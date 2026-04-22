import os
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from .utils.v3dc_io import read_v3dc_sliced
from .depth_estimators import setup_depth  # <-- you need this


class GeneratorV3DC:

    def __init__(self, cfg, v3dc_path):
        self.cfg = cfg
        self.v3dc_path = v3dc_path
        self.outputs = []

        print(f"Loading V3DC from {v3dc_path}")

        self.views = read_v3dc_sliced(
            fp=v3dc_path,
            start=0,
            end=None,
            read_rgb=True,
            read_depth=True,
            orient_img=True,
        )

        print(f"Loaded {len(self.views)} frames")

        # ✅ NEW: depth estimator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_estimator = setup_depth(self.cfg.depth_estimator)

    def is_blurry(self, image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold, laplacian_var

    def process_views(self):

        for i, view in enumerate(tqdm(self.views)):
            rgb = view["img"]
            depth_scan = view["depth"]
            pose = view["viewmat"]

            if rgb is None:
                continue

            blurry, score = self.is_blurry(rgb)
            if blurry:
                print(f"Skipping blurry frame {i}, score={score:.2f}")
                continue

            # -------------------------
            # Resize scan depth to RGB
            # -------------------------
            if depth_scan is not None:
                depth_scan = cv2.resize(
                    depth_scan,
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_scan = depth_scan.astype(np.float32) / 1000.0  # meters
            else:
                depth_scan = np.zeros(rgb.shape[:2], dtype=np.float32)

            # -------------------------
            # VALID DEPTH MASK (IMPORTANT)
            # -------------------------
            valid_scan = (depth_scan > 0.0) & (depth_scan < 10.0)

            # -------------------------
            # MONOCULAR DEPTH (FILL GAPS)
            # -------------------------
            rgb = np.ascontiguousarray(rgb)

            rgb_tensor = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255.0
            )
            

            with torch.no_grad():
                depth_pred = self.depth_estimator(rgb_tensor)

            depth_pred = depth_pred[0, 0].cpu().numpy()

            # Normalize predicted depth
            depth_pred = depth_pred / (depth_pred.max() + 1e-6)

            # -------------------------
            # COMBINE DEPTHS (KEY STEP)
            # -------------------------
            depth = np.where(valid_scan, depth_scan, depth_pred)

            # -------------------------
            # CROP CENTER
            # -------------------------
            H, W = rgb.shape[:2]
            min_dim = min(H, W)

            sx = (W - min_dim) // 2
            sy = (H - min_dim) // 2

            rgb = rgb[sy:sy+min_dim, sx:sx+min_dim]
            depth = depth[sy:sy+min_dim, sx:sx+min_dim]
            valid_scan = valid_scan[sy:sy+min_dim, sx:sx+min_dim]

            # -------------------------
            # RESIZE TO MODEL SIZE
            # -------------------------
            rgb = cv2.resize(rgb, (self.cfg.img_size, self.cfg.img_size))
            depth = cv2.resize(depth, (self.cfg.img_size, self.cfg.img_size))
            valid_scan = cv2.resize(
                valid_scan.astype(np.uint8),
                (self.cfg.img_size, self.cfg.img_size),
                interpolation=cv2.INTER_NEAREST
            )

            # -------------------------
            # FINAL MASK (CRITICAL FIX)
            # -------------------------
            # Mask ONLY where NO reliable geometry
            mask = (~valid_scan).astype(np.uint8)

            # -------------------------
            # FORMAT FOR TRAINING
            # -------------------------
            rgb = np.transpose(rgb, (2, 0, 1))[None]
            depth = depth[None, None]
            mask = mask[None, None]

            self.outputs.append({
                "pose_index": i,
                "pose": pose,
                "rgb": rgb,
                "depth": depth,
                "mask": mask
            })

        print(f"Processed {len(self.outputs)} frames")

    def export_to_dataset(self):

        output_path = os.path.join(self.cfg.output_path, self.cfg.scene_name)
        os.makedirs(output_path, exist_ok=True)

        nerfstudio_json = {
            "camera_model": "OPENCV",
            "fl_x": self.cfg.img_size / 2,
            "fl_y": self.cfg.img_size / 2,
            "cx": self.cfg.img_size / 2,
            "cy": self.cfg.img_size / 2,
            "w": self.cfg.img_size,
            "h": self.cfg.img_size,
            "frames": [],
        }

        for out in tqdm(self.outputs):
            idx = out["pose_index"]
            pose = out["pose"]

            rgb = out["rgb"][0].transpose(1, 2, 0)
            depth = out["depth"][0, 0]
            mask = 1 - out["mask"][0, 0].astype(np.uint8)

            rgb_path = os.path.join(output_path, "rgb", f"{idx}.png")
            depth_path = os.path.join(output_path, "depth", f"{idx}.npy")
            mask_path = os.path.join(output_path, "mask", f"{idx}.png")

            for path in [rgb_path, depth_path, mask_path]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
            np.save(depth_path, depth)

            # Depth visualization
            depth_vis = depth / (depth.max() + 1e-6)
            plt.imshow(depth_vis, cmap="turbo")
            plt.colorbar()
            plt.savefig(depth_path.replace(".npy", ".png"))
            plt.close()

            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            transform = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])

            pose_gl = transform @ pose
            pose_gl_c2w = np.linalg.inv(pose_gl)

            nerfstudio_json["frames"].append({
                "file_path": os.path.join("rgb", f"{idx}.png"),
                "depth_file_path": os.path.join("depth", f"{idx}.npy"),
                "mask_path": os.path.join("mask", f"{idx}.png"),
                "mask_inpainting_file_path": os.path.join("mask", f"{idx}.png"),
                "transform_matrix": pose_gl_c2w.tolist()
            })

        with open(os.path.join(output_path, "init_transforms.json"), "w") as f:
            json.dump(nerfstudio_json, f, indent=4)

        print(f"Dataset exported to {output_path}")