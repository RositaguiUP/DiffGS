import argparse
import os
import json
import numpy as np
from rich import print
from pathlib import Path
import time

from cprint import cprint

from configs import get_cfg_defaults
from pcd_generator.pcd_renderer import PCDRenderer

def load_poses_from_json(json_path):
    """
    Loads camera poses from a JSON file in the format of views.json
    Returns a list of 4x4 numpy arrays.
    """
    with open(json_path, 'r') as f:
        views = json.load(f)

    poses = []
    for v in views:
        viewmat = np.array(v['viewmat'], dtype=np.float32)
        poses.append(viewmat)
    return poses

def main(cfg):
    cprint.info("Starting Experiment: {}".format(cfg.project_name))
    cprint.info("Starting at: {}".format(time.ctime()))

    # Paths from config
    input_pcd_path = cfg.pointcloud_path  # path to the existing point cloud (PLY)
    views_json_path = cfg.views_json_path  # path to views.json
    output_path = cfg.output_path

    if not Path(input_pcd_path).exists():
        raise FileNotFoundError(f"Point cloud file not found: {input_pcd_path}")
    if not Path(views_json_path).exists():
        raise FileNotFoundError(f"Views JSON file not found: {views_json_path}")

    print("[bold blue]Loading camera poses...[/bold blue]")
    poses = load_poses_from_json(views_json_path)
    
    print("[bold blue]Initializing PCD Renderer...[/bold blue]")
    renderer = PCDRenderer(cfg, input_pcd_path=input_pcd_path, poses=poses)
    
    print("[bold blue]Rendering from loaded poses...[/bold blue]")
    renderer.render_from_poses()

    print("[bold blue]Exporting dataset...[/bold blue]")
    renderer.export_to_dataset()

    print(f"[bold green]Done! Dataset exported to {output_path}[/bold green]")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config_path", help="Path to Config File", required=True, default=""
    )
    args, _ = args.parse_known_args()

    # Load config file
    cfg = get_cfg_defaults()
    
    if (
        os.path.exists(args.config_path)
        and os.path.splitext(args.config_path)[1] == ".yaml"
    ):
        cfg.merge_from_file(args.config_path)
    else:
        print("No valid config specified")
        exit(1)

    cprint.info(cfg)
    
    main(cfg)