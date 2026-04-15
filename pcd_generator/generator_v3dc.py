import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from .utils.v3dc_io import read_v3dc_sliced


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
        
    def is_blurry(self, image, threshold=100.0):
        """
        Returns True if image is blurry. Variance of Laplacian
        threshold: lower = stricter (more images removed)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold, laplacian_var

    def process_views(self):
        """
        Extract RGB, depth, mask, and pose from V3DC
        """

        for i, view in enumerate(tqdm(self.views)):
            rgb = view["img"]
            depth = view["depth"]
            pose = view["viewmat"]  # camera pose

            if rgb is None or depth is None:
                continue
            
            blurry, score = self.is_blurry(rgb, threshold=100.0)

            if blurry:
                print(f"Skipping blurry frame {i}, score={score:.2f}")
                continue
            
            # Step 1: resize depth to RGB resolution FIRST
            depth = cv2.resize(
                depth,
                (rgb.shape[1], rgb.shape[0]),  # (W, H)
                interpolation=cv2.INTER_NEAREST
            )

            # Step 2: now shapes match
            H, W = rgb.shape[:2]
            min_dim = min(H, W)

            start_x = (W - min_dim) // 2
            start_y = (H - min_dim) // 2

            # Step 3: crop BOTH
            rgb = rgb[start_y:start_y+min_dim, start_x:start_x+min_dim]
            depth = depth[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # Step 4: resize to model input
            rgb = cv2.resize(rgb, (self.cfg.img_size, self.cfg.img_size))
            depth = cv2.resize(depth, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)
            
            mask = (depth <= 0).astype(np.uint8)
            
            # print(rgb.shape, depth.shape, mask.shape)
            
            # THEN convert to torch format
            rgb = np.transpose(rgb, (2, 0, 1))[None]      # (1,3,H,W)
            depth = (depth.astype(np.float32) / 1000.0)[None, None]  # (1,1,H,W)
            mask = mask[None, None]                       # (1,1,H,W)
            

            self.outputs.append({
                "pose_index": i,
                "pose": pose,
                "rgb": rgb,
                "depth": depth,
                "mask": mask
            })

        print(f"Processed {len(self.outputs)} frames")

    def export_to_dataset(self):
        """
        Save dataset in Nerfstudio format
        """

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

            rgb = out["rgb"][0].transpose(1, 2, 0)  # CHW → HWC
            depth = out["depth"][0, 0]
            mask = 1 - out["mask"][0, 0].astype(np.uint8)

            # Paths
            rgb_path = os.path.join(output_path, "rgb", f"{idx}.png")
            depth_path = os.path.join(output_path, "depth", f"{idx}.npy")
            mask_path = os.path.join(output_path, "mask", f"{idx}.png")

            for path in [rgb_path, depth_path, mask_path]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save RGB
            Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)

            # Save depth
            np.save(depth_path, depth)

            # Optional depth visualization
            depth_vis = depth / (depth.max() + 1e-6)
            plt.imshow(depth_vis, cmap="turbo")
            plt.colorbar()
            plt.savefig(depth_path.replace(".npy", ".png"))
            plt.close()

            # Save mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            # Convert pose to OpenGL (same as your original)
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

        # Save JSON
        with open(os.path.join(output_path, "init_transforms.json"), "w") as f:
            json.dump(nerfstudio_json, f, indent=4)

        print(f"Dataset exported to {output_path}")