import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from rich import print
from tqdm.auto import tqdm
import json

from .pcd_handler import setup_renderer


class PCDRenderer:

    def __init__(self, cfg, input_pcd_path=None, poses=[]):
        self.cfg = cfg
        self.input_pcd_path = input_pcd_path
        self.renderer = setup_renderer(cfg.pcd_renderer)
        self.poses = poses
        
        self.outputs = []

        if input_pcd_path is not None:
            self.renderer.load_from_ply(input_pcd_path)
            
            # init_pose = np.eye(4)
            init_pose = poses[0]
            self.renderer.init_shadow_volume(pose=init_pose)
            
            print(f"Loaded PCD from {input_pcd_path}")
        else:
            raise ValueError("You must provide an input PCD path.")
        
        
    def render_from_poses(self):
        """
        Render RGB, depth, mask from a list of 4x4 camera poses.
        Each pose should be a numpy array (4x4).
        """

        for i, pose in enumerate(tqdm(self.poses)):
            rgb, depth, ids = self.renderer(pose, with_shadow_volume=False)
            rgb_w_shadow, depth_w_shadow, ids_w_shadow = self.renderer(
                pose, with_shadow_volume=True
            )

            mask_empty = depth < 0

            # Save outputs in memory or write immediately
            self.outputs.append({
                "pose_index": i,
                "pose": pose,
                "rgb": rgb.cpu(),
                "depth": depth.cpu(),
                "mask": mask_empty.cpu()
            })

        print(f"[bold green]Rendered {len(self.poses)} poses from existing PCD![/bold green]")

    def export_to_dataset(self):
        """
        Save rendered frames (RGB, depth, mask) and poses to disk
        """
        # Create the directory for this scene
        output_path = os.path.join(self.cfg.output_path, self.cfg.scene_name)
        if not os.path.exists(output_path):
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
            rgb = transforms.ToPILImage()(out["rgb"][0])
            depth = out["depth"][0, 0].numpy() * 1000
            mask = 1 - out["mask"][0, 0].numpy()

            # Paths
            rgb_path = os.path.join(output_path, "rgb", f"{idx}.png")
            depth_path = os.path.join(output_path, "depth", f"{idx}.npy")
            mask_path = os.path.join(output_path, "mask", f"{idx}.png")

            for path in [rgb_path, depth_path, mask_path]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            rgb.save(rgb_path)
            np.save(depth_path, depth)
            plt.imshow(depth / 1000, cmap="turbo")
            plt.colorbar()
            plt.savefig(depth_path.replace(".npy", ".png"))
            plt.close()
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            # Transform pose for nerfstudio
            transform = np.array([[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, 0, 0, 1]])
            pose_gl = transform @ pose
            pose_gl_c2w = np.linalg.inv(pose_gl)

            nerfstudio_json["frames"].append({
                "file_path": os.path.join("rgb", f"{idx}.png"),
                "depth_file_path": os.path.join("depth", f"{idx}.npy"),
                "mask_path": os.path.join("mask", f"{idx}.png"),
                "mask_inpainting_file_path": os.path.join("mask", f"{idx}.png"),
                "transform_matrix": pose_gl_c2w.tolist()
            })

        # Save nerfstudio JSON
        with open(os.path.join(output_path, "init_transforms.json"), "w") as f:
            json.dump(nerfstudio_json, f, indent=4)

        # Save PCD as ply
        self.renderer.write_to_ply(os.path.join(output_path, "pointcloud.ply"))

        print(f"[bold green]Export complete at {output_path}![/bold green]")