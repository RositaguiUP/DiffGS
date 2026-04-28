import os
import json
import numpy as np
import cv2
import torch
import kornia

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from kornia.enhance import sharpness

from .utils.v3dc_io import read_v3dc_sliced


class GeneratorV3DC:
    def __init__(self, cfg, v3dc_path, mesh_depth_path):
        self.cfg = cfg
        self.v3dc_path = v3dc_path
        self.mesh_depth_path = mesh_depth_path
        self.outputs = []

        print(f"Loading V3DC from {v3dc_path}")

        self.views = read_v3dc_sliced(
            fp=v3dc_path,
            start=0,
            end=None,
            step=1,
            read_rgb=True,
            read_depth=True,
            orient_img=True,
        )

        print(f"Loaded {len(self.views)} frames")

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    def process_views(self, combine_depths_for_mask=True):
        for i, view in enumerate(tqdm(self.views)):
            processed = self.process_single_view(view, combine_depths_for_mask)

            if processed is not None:
                self.outputs.append(processed)

        print(f"Processed {len(self.outputs)} frames")

    def process_single_view(self, view, combine_depths_for_mask):
        idx = view["idx"]
        rgb = view["img"]
        scan_depth = view["depth"]
        pose = view["viewmat"]

        if rgb is None or scan_depth is None:
            return None

        scan_depth = scan_depth.astype(np.float32) / 1000.0

        if self.should_skip_blurry(rgb, idx, threshold=20):
            return None

        mesh_depth = self.load_mesh_depth(idx)
        if self.mesh_depth_path and mesh_depth is None:
            print(f'No mesh depth {idx}')
            return None

        rgb, scan_depth = self.preprocess_rgb_depth(rgb, scan_depth)

        if mesh_depth is not None:
            mesh_depth = self.resize_depth(mesh_depth, self.cfg.img_size)

        mask = self.create_mask(
            scan_depth,
            mesh_depth,
            combine_depths_for_mask
        )

        rgb = self.apply_optional_sharpness(rgb)

        return self.to_output_dict(idx, pose, rgb, scan_depth, mask)

    # ------------------------------------------------------------------
    # IMAGE QUALITY
    # ------------------------------------------------------------------

    def should_skip_blurry(self, rgb, idx, threshold=100.0):
        blurry, score = self.is_blurry(rgb, threshold)

        if blurry:
            print(f"Skipping blurry frame {idx}, score={score:.2f}")

        return blurry

    def is_blurry(self, image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < threshold, score

    # ------------------------------------------------------------------
    # DEPTH LOADING
    # ------------------------------------------------------------------

    def load_mesh_depth(self, idx):
        if not self.mesh_depth_path:
            return None

        path = Path(self.mesh_depth_path) / f"{idx}.npy"

        if not path.exists():
            return None

        return np.load(path)

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------

    def preprocess_rgb_depth(self, rgb, depth):
        depth = self.resize_depth(depth, rgb.shape[1], rgb.shape[0])

        rgb, depth = self.center_crop_square(rgb, depth)

        rgb = cv2.resize(rgb, (self.cfg.img_size, self.cfg.img_size))
        depth = self.resize_depth(depth, self.cfg.img_size)

        return rgb, depth

    def resize_depth(self, depth, w, h=None):
        if h is None:
            h = w

        return cv2.resize(
            depth,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    def center_crop_square(self, rgb, depth):
        h, w = rgb.shape[:2]
        size = min(h, w)

        start_x = (w - size) // 2
        start_y = (h - size) // 2

        rgb = rgb[start_y:start_y + size, start_x:start_x + size]
        depth = depth[start_y:start_y + size, start_x:start_x + size]

        return rgb, depth

    # ------------------------------------------------------------------
    # MASKS
    # ------------------------------------------------------------------

    def create_mask(self, scan_depth, mesh_depth, combine_depths):
        mask_scan_gap = scan_depth <= 0

        if mesh_depth is None:
            return mask_scan_gap.astype(np.uint8)

        mask_mesh_gap = mesh_depth <= 0

        if combine_depths:
            mask = mask_scan_gap | mask_mesh_gap
        else:
            mask = mask_mesh_gap

        return mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # SHARPNESS FILTER
    # ------------------------------------------------------------------

    def apply_optional_sharpness(self, rgb):
        use_sharpness = self.cfg.use_sharpness

        if not use_sharpness:
            return rgb

        factor = self.cfg.sharpness_factor

        tensor = (
            torch.from_numpy(rgb)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0) / 255.0
        )

        tensor = sharpness(tensor, factor=factor)

        rgb = (
            tensor[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

        return rgb

    # ------------------------------------------------------------------
    # FORMAT OUTPUT
    # ------------------------------------------------------------------

    def to_output_dict(self, idx, pose, rgb, depth, mask):
        rgb = np.transpose(rgb, (2, 0, 1))[None]
        depth = depth[None, None]
        mask = mask[None, None]

        return {
            "pose_index": idx,
            "pose": pose,
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
        }

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def export_to_dataset(self):
        output_path = os.path.join(
            self.cfg.output_path,
            self.cfg.scene_name
        )

        os.makedirs(output_path, exist_ok=True)

        transforms = self.create_transforms_template()

        for out in tqdm(self.outputs):
            self.export_frame(output_path, transforms, out)

        json_path = os.path.join(output_path, "init_transforms.json")

        with open(json_path, "w") as f:
            json.dump(transforms, f, indent=4)

        print(f"Dataset exported to {output_path}")

    def create_transforms_template(self):
        s = self.cfg.img_size

        return {
            "camera_model": "OPENCV",
            "fl_x": s / 2,
            "fl_y": s / 2,
            "cx": s / 2,
            "cy": s / 2,
            "w": s,
            "h": s,
            "frames": [],
        }

    def export_frame(self, output_path, transforms, out):
        idx = out["pose_index"]
        pose = out["pose"]

        rgb = out["rgb"][0].transpose(1, 2, 0)
        depth = out["depth"][0, 0]
        mask = 1 - out["mask"][0, 0].astype(np.uint8)

        rgb_path = os.path.join(output_path, "rgb", f"{idx}.png")
        depth_path = os.path.join(output_path, "depth", f"{idx}.npy")
        mask_path = os.path.join(output_path, "mask", f"{idx}.png")

        self.ensure_parent_dirs(rgb_path, depth_path, mask_path)

        Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
        np.save(depth_path, depth)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

        self.save_depth_visualization(depth, depth_path)

        pose_c2w = self.convert_pose_to_opengl(pose)

        transforms["frames"].append({
            "file_path": os.path.join("rgb", f"{idx}.png"),
            "depth_file_path": os.path.join("depth", f"{idx}.npy"),
            "mask_path": os.path.join("mask", f"{idx}.png"),
            "mask_inpainting_file_path": os.path.join("mask", f"{idx}.png"),
            "transform_matrix": pose_c2w.tolist(),
        })

    def ensure_parent_dirs(self, *paths):
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def save_depth_visualization(self, depth, depth_path):
        depth_vis = depth / (depth.max() + 1e-6)

        plt.imshow(depth_vis, cmap="turbo")
        plt.colorbar()
        plt.savefig(depth_path.replace(".npy", ".png"))
        plt.close()

    def convert_pose_to_opengl(self, pose):
        transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        pose_gl = transform @ pose
        pose_c2w = np.linalg.inv(pose_gl)

        return pose_c2w