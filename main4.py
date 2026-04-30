import argparse
import os
import json
import numpy as np
from rich import print
from pathlib import Path
import time

from cprint import cprint

from configs import get_cfg_defaults
from inputs_generator.generator import Generator

def main(cfg):
    cprint.info("Starting Experiment: {}".format(cfg.project_name))
    cprint.info("Starting at: {}".format(time.ctime()))

    # Paths from config
    output_path = Path(cfg.output_path) / cfg.scene_name
    
    print("[bold blue]Initializing Generator...[/bold blue]")
    generator = Generator(cfg.env_id, cfg.floor_number, output_path)
    
    generator.run(cfg.img_size, cfg.dist_thresh, cfg.rot_thresh, cfg.blur_thresh, cfg.step)

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