# src/renderer_mesh_depth.py

from pathlib import Path
import numpy as np
import torch

from inputs_generator.src.environment import Environment

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    MeshRenderer,
    HardPhongShader,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Renderer:
    def __init__(self, env_id: str, floor_number: int):
        self.env = Environment(env_id)
        self.floor_number = floor_number

        obj_path = self.env.path_mesh(floor_number) / "model.obj"

        self.mesh = load_objs_as_meshes(
            [obj_path],
            create_texture_atlas=True,
            device=device
        )

    def meshes_batch(self, batch_size):
        if batch_size == 1:
            return self.mesh
        return join_meshes_as_batch(
            [self.mesh for _ in range(batch_size)]
        )

    def extrinsics_to_cameras(
        self,
        extrinsics,
        intrinsics,
        image_size,
        scale=1.0
    ):
        extrinsics = np.asarray(extrinsics)
        intrinsics = np.asarray(intrinsics)

        focal = np.stack([
            intrinsics[..., 0, 0],
            intrinsics[..., 1, 1]
        ], axis=-1) * scale

        pp = np.stack([
            intrinsics[..., 0, 2],
            intrinsics[..., 1, 2]
        ], axis=-1) * scale

        H = int(round(image_size[0] * scale))
        W = int(round(image_size[1] * scale))

        w2c = extrinsics

        R = np.transpose(w2c[..., :3, :3], (0, 2, 1))
        T = w2c[..., :3, 3]

        # PyTorch3D convention
        R[..., :2] *= -1
        T[..., :2] *= -1

        cams = PerspectiveCameras(
            focal_length=torch.tensor(
                focal, dtype=torch.float32, device=device
            ),
            principal_point=torch.tensor(
                pp, dtype=torch.float32, device=device
            ),
            R=torch.tensor(
                R, dtype=torch.float32, device=device
            ),
            T=torch.tensor(
                T, dtype=torch.float32, device=device
            ),
            image_size=torch.tensor(
                [[H, W]], dtype=torch.float32, device=device
            ),
            in_ndc=False,
            device=device,
        )

        return cams

    @torch.no_grad()
    def render_rgb_depth(self, cameras, imsize):
        raster_settings = RasterizationSettings(
            image_size=imsize,
            faces_per_pixel=1,
            blur_radius=0.0,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(
                device=device,
                cameras=cameras
            )
        )

        mesh = self.meshes_batch(len(cameras))

        images = renderer(mesh)
        fragments = rasterizer(mesh)

        rgb = (
            images[0, ..., :3]
            .cpu()
            .numpy()
        )

        rgb = (255 * rgb).astype(np.uint8)

        depth = fragments.zbuf[0, ..., 0].cpu().numpy()

        # invalid pixels -> 0
        depth[np.isinf(depth)] = 0
        depth[depth < 0] = 0

        return rgb, depth