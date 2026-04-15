import os
import json
import torch
import imageio
import argparse
from pathlib import Path

# --- YOUR IMPORTS (adapt paths) ---
from realmdreamer.realmdreamer.gaussian_splatting.gs_model import GaussianSplatting
from realmdreamer.realmdreamer.gaussian_splatting.gs_camera_utils import ns2gs_camera

from nerfstudio.nerfstudio.utils.eval_utils import eval_setup

from nerfstudio.nerfstudio.cameras.cameras import Cameras


def build_camera(camera_dict, device):
    c2w = torch.tensor(camera_dict["camera_to_world"], dtype=torch.float32)
    
    if c2w.numel() == 16:
        c2w = c2w.view(4, 4)[:3, :4]
    else:
        raise ValueError(f"Unexpected camera_to_world shape: {c2w.shape}")

    height = camera_dict.get("render_height", 512)
    width = camera_dict.get("render_width", 512)
    fov = camera_dict["fov"]

    fx = 0.5 * width / torch.tan(torch.deg2rad(torch.tensor(fov)) / 2)
    fy = fx

    from nerfstudio.cameras.cameras import Cameras

    camera = Cameras(
        camera_to_worlds=c2w.unsqueeze(0).to(device),  # [1,3,4]
        fx=fx.unsqueeze(0).to(device),
        fy=fy.unsqueeze(0).to(device),
        cx=torch.tensor([width / 2], device=device),
        cy=torch.tensor([height / 2], device=device),
        width=torch.tensor([width], device=device),
        height=torch.tensor([height], device=device),
    )

    return camera

def load_model(config_path):
    # # Load config if needed (yaml → dict)
    # import yaml
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)

    # # Init model
    # model = GaussianSplatting(config=config)
    # model.to(device)

    # # Load checkpoint
    # ckpt = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(ckpt["model"], strict=False)

    # model.eval()
  

    config, pipeline, checkpoint_path, _ = eval_setup(
        Path(config_path),
        eval_num_rays_per_chunk=None,
        test_mode="inference"
    )

    model = pipeline.model
    model.eval()
    
    return model


@torch.no_grad()
def render_path(model, camera_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    for i, cam_dict in enumerate(camera_path):
        camera = build_camera(cam_dict, device)

        # Convert to GS camera
        viewpoint_camera = ns2gs_camera(camera, device=device)

        # Forward pass
        outputs = model.forward(viewpoint_camera)

        rgb = outputs["rgb"][0].cpu().numpy()
        depth = outputs["depth"][0].cpu().numpy()

        # Save
        imageio.imwrite(os.path.join(output_dir, f"{i:04d}.png"), (rgb * 255).astype("uint8"))
        imageio.imwrite(os.path.join(output_dir, f"{i:04d}_depth.png"), depth.squeeze())

        print(f"Rendered frame {i}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--camera_path", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args.config)

    # Load camera path
    with open(args.camera_path, "r") as f:
        camera_data = json.load(f)

    camera_path = camera_data["camera_path"]

    # Render
    render_path(model, camera_path, args.output, device)


if __name__ == "__main__":
    main()