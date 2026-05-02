import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler
)

class SDControlNetConfig:
    # pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tile_controlnet_path: str = "lllyasviel/control_v11f1e_sd15_tile"
    depth_controlnet_path: str = "lllyasviel/control_v11f1p_sd15_depth"
    
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: list =[1.0, 1.0] # [Tile, Depth]
    
    min_step_percent: float = 0.05
    max_step_percent: float = 0.60  # Max 60% noise to preserve room structure
    
    num_steps_sample: int = 20

class StableDiffusionControlNetGuidance(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.cfg = SDControlNetConfig()
        self.device = device
        self.weights_dtype = torch.float16 # FP16 for speed/VRAM

        print("Loading ControlNets...")
        controlnet_tile = ControlNetModel.from_pretrained(
            self.cfg.tile_controlnet_path, torch_dtype=self.weights_dtype
        )
        controlnet_depth = ControlNetModel.from_pretrained(
            self.cfg.depth_controlnet_path, torch_dtype=self.weights_dtype
        )
        
        print("Loading Stable Diffusion ControlNet Pipeline...")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=[controlnet_tile, controlnet_depth],
            torch_dtype=self.weights_dtype,
            safety_checker=None,
        ).to(self.device)

        # Optimize VRAM
        self.pipe.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler
        self.pipe.set_progress_bar_config(disable=True)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

    @torch.no_grad()
    def multi_step(
        self,
        rgb,                # B H W C (Sharp 3DGS Render)
        scan_rgb,           # B H W C (Blurry GT Scan)
        scan_depth,         # B 1 H W (Smooth GT Scan Depth)
        prompt: str,
        current_step_ratio: float,
    ):
        batch_size = rgb.shape[0]
        original_h, original_w = rgb.shape[1], rgb.shape[2]
        
        # 1. Prepare Base Latent Image (The 3DGS Render)
        rgb_BCHW = rgb.permute(0, 3, 1, 2).clamp(0, 1)
        if rgb_BCHW.shape[-2:] != (512, 512):
            rgb_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)
        else:
            rgb_512 = rgb_BCHW
        
        # 2. Prepare ControlNet Inputs
        # Tile ControlNet needs the Scan RGB
        ctrl_tile = scan_rgb.permute(0, 3, 1, 2).clamp(0, 1) 
        if ctrl_tile.shape[-2:] != (512, 512):
            ctrl_tile_512 = F.interpolate(ctrl_tile, (512, 512), mode="bilinear", align_corners=False)
        else:
            ctrl_tile_512 = ctrl_tile
        
        # Depth ControlNet needs a 3-channel normalized depth map
        # Normalize depth to 0-1 (Ignore 0s for Min/Max!)
        valid_mask = scan_depth > 0
        if valid_mask.sum() > 0:
            d_min, d_max = scan_depth[valid_mask].min(), scan_depth[valid_mask].max()
        else:
            d_min, d_max = 0.0, 1.0
        
        ctrl_depth = torch.where(
            valid_mask,
            (scan_depth - d_min) / (d_max - d_min + 1e-8),
            torch.zeros_like(scan_depth)
        )
        ctrl_depth = ctrl_depth.repeat(1, 3, 1, 1).clamp(0, 1) # B 3 H W
        if ctrl_depth.shape[-2:] != (512, 512):
            ctrl_depth_512 = F.interpolate(ctrl_depth, (512, 512), mode="nearest")
        else:
            ctrl_depth_512 = ctrl_depth
        
        # 3. Anneal the Timestep (Start high noise, end low noise)
        t = current_step_ratio * self.min_step + (1 - current_step_ratio) * self.max_step
        strength = t / self.num_train_timesteps

        # 4. Generate the Perfect "Pseudo-GT" Image  (Runs in FP16)
        # We pass the Sharp Render as the starting image.
        out_images = self.pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=["blurry, motion blur, out of focus, distorted, artifact, worst quality"] * batch_size,
            image=rgb_512.to(self.weights_dtype), 
            control_image=[ctrl_tile_512.to(self.weights_dtype), ctrl_depth_512.to(self.weights_dtype)],
            controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
            strength=strength,
            num_inference_steps=self.cfg.num_steps_sample,
            guidance_scale=self.cfg.guidance_scale,
            output_type="pt"
        ).images

        # out_images is B C H W
        pseudo_gt_sharp = F.interpolate(out_images, (original_h, original_w), mode="bilinear", align_corners=False)
        
        return pseudo_gt_sharp.clamp(0, 1).to(torch.float32)