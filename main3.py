import argparse
import os
import json
import numpy as np
from rich import print
from pathlib import Path
import time

from cprint import cprint

from configs import get_cfg_defaults
from pcd_generator.generator_v3dc import GeneratorV3DC
# from pcd_generator.generator_v3dc2 import GeneratorV3DC

def main(cfg):
    cprint.info("Starting Experiment: {}".format(cfg.project_name))
    cprint.info("Starting at: {}".format(time.ctime()))

    # Paths from config
    v3dc_path = cfg.v3dc_path
    output_path = cfg.output_path

    if not Path(v3dc_path).exists():
        raise FileNotFoundError(f"Point cloud file not found: {v3dc_path}")
    
    mesh_depth_path = Path("/home/rosita/tests/rendering/mesh_rendering/outputs/f1/depth")
    
    print("[bold blue]Initializing Generator from V3DC...[/bold blue]")
    generator = GeneratorV3DC(cfg, v3dc_path=v3dc_path, mesh_depth_path=mesh_depth_path)
    
    print("[bold blue]Processing views...[/bold blue]")
    generator.process_views()

    print("[bold blue]Exporting dataset...[/bold blue]")
    generator.export_to_dataset()

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