from __future__ import annotations

import json
import math
import os
import pdb
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Tuple, Type, Union

import kornia
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)
from jaxtyping import Bool, Float
from kornia.color.colormap import AUTUMN
import kornia.losses as k_losses
from kornia.enhance import sharpness
from kornia.filters import gaussian_blur2d, laplacian
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import (
    DepthLossType, L1Loss, MSELoss, ScaleAndShiftInvariantLoss, depth_loss,
    distortion_loss, interlevel_loss, lossfun_outer, orientation_loss,
    pred_normal_loss, ray_samples_to_sdist,
    scale_gradients_by_distance_squared, tv_loss)
from nerfstudio.model_components.ray_samplers import (ProposalNetworkSampler,
                                                      UniformSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   NormalsRenderer,
                                                   RGBRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import ViewerSlider
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.image import (MultiScaleStructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.depth import (depth_pearson_loss, depth_ranking_loss,
                         depth_ranking_loss_multi_patch_masked,
                         patch_pearson_loss)

from realmdreamer.guidance.geowizard_guidance import GeoWizardGuidance
from realmdreamer.guidance.marigold_guidance import (MarigoldGuidance,
                                                     align_depths)
from realmdreamer.guidance.sd_guidance import StableDiffusionGuidance
from realmdreamer.guidance.sd_inpainting_guidance import \
    StableDiffusionInpaintingGuidance

from .gs_camera import Camera as GaussianSplattingCamera
from .gs_camera_utils import ns2gs_camera
from .gs_field import GaussianSplattingField, GaussianSplattingFieldConfig
from .gs_graphics_utils import focal2fov, fov2focal, getWorld2View2
from .gs_sh_utils import eval_sh
from .gs_utils import wide_sigmoid

# from realmdreamer.guidance.sd_inpainting_guidance_ism import \
# StableDiffusionInpaintingISMGuidance




@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: GaussianSplatting)

    gaussian_model: GaussianSplattingFieldConfig = GaussianSplattingFieldConfig()
    """Config for the Gaussian model."""
    
    # === Mode Switch ===
    is_distillation: bool = False
    """If True, runs ControlNet distillation. If False, runs standard 3DGS reconstruction."""
    
    deblur_enabled: bool = False
    """Whether to enable per-image learnable deblurring kernels."""
    
    deblur_kernel_size: int = 15
    """Size of the square blur kernel."""

    background_color: str = "white"
    """Background color for the rendered images. Either "black" or "white"."""

    sh_degree: int = 0
    """Maximum SH degree to use for rendering."""

    pcd_path: str = None
    """Path to the model to load."""

    full_precision: bool = False
    """Whether to use full precision for SDS guidance."""

    guidance: str = "sds_inpainting"
    """sds or sds_inpainting"""

    guidance_scale: float = 100.0
    """Guidance scale for SDS guidance."""

    img_guidance_scale: float = 1.0
    """Guidance scale for image conditioning in Inpainting NFSD/VSD"""

    min_step_percent: float = 0.02
    """Minimum step percent for SDS guidance."""

    max_step_percent: float = 0.98
    """Maximum step percent for SDS guidance."""

    ignore_mask: bool = False
    """Whether or not to ignore the inpainting mask"""

    anneal: bool = False
    """Whether or not to anneal timesteps"""

    prolific_anneal: bool = False
    """Whether or not to anneal timesteps according to prolific dreamer"""

    loss_type: str = "noise"
    """Either noise, one_step, or multi_step, or nfsd - determines if regular SDS, or equivalent denoised SDS or multi-step SDS (Ref Sparsefusion) - Only supports sd_inpainting"""

    invert_ddim: bool = False
    """Whether to use DDIM inversion for computing noisy latents"""

    invert_after_step: bool = False
    """Use DDIM inversion only after a particular step"""

    invert_step_ratio: float = 0.1
    """Step to start using DDIM inversion"""

    ddim_invert_method: str = "ddim" or "pseudo_ddim" or "ddim_0"
    """Whether to use DDIM inversion for computing noisy latents or DDPM or pseudo ddim"""

    num_steps_sample: int = 10
    """Number of steps to sample for multi-step SDS"""

    fixed_num_steps: bool = False
    """Whether to use a fixed number of steps for multi-step SDS"""

    set_sds_weight_l2_and_perceptual: bool = False
    """Whether to set the weight of the SDS loss (from the alpha) for the l2 and perceptual loss"""

    lambda_sds: float = 0.1
    """Multiplier for SDS loss."""

    lambda_rgb: float = 1000.0
    """Multiplier for RGB loss."""

    lambda_depth: float = 0.0
    """Multiplier for depth loss."""

    lambda_opaque: float = 0.0
    """Multiplier for opaqueness loss."""

    lambda_one_step: float = 0.0
    """Multiplier for RGB loss (l2) with one step predictions"""

    lambda_one_step_l1: float = 0.0
    """Multiplier for L1 loss with one step predictions"""

    lambda_one_step_perceptual: float = 0.0
    """Multiplier for perceptual loss with one step predictions"""

    lambda_one_step_ssim: float = 0.0
    """Multiplier for SSIM loss with one step predictions"""

    lambda_input_constraint_l2: float = 0.0
    """Multiplier for the anchor loss on the input view"""

    lambda_input_constraint_perceptual: float = 0.0
    """Multiplier for the perceptual loss on the input view"""

    lambda_input_constraint_depth: float = 0.0
    """Multiplier for the depth loss on the input view"""

    use_sigmoid: bool = False
    """Whether to use sigmoid on color activations"""

    average_colors: bool = False
    """Average color of that are near each other before rendering"""

    depth_guidance: bool = False
    """Whether to use depth guidance for SDS"""

    depth_guidance_multi_step: bool = True
    """Whether to use multi-step depth guidance for SDS"""

    load_depth_guidance: bool = False
    """Whether to load depth guidance model for the diffusion model - can be used if even if depth_guidance is False"""

    lambda_depth_sds: float = 100.0
    """The relative weight to the first SDS to use for depth SDS"""

    depth_loss: Literal["pearson", "patch_pearson", "ranking", "ranking_multi_patch"] = "pearson"
    """If using depth loss, whether to use MSE (l2), ranking, or pearson loss or patch pearson loss"""

    depth_patch_size: int = 64
    """Patch size for patch pearson/ranking loss"""

    depth_num_patches: int = 64
    """Number of patches to use for ranking loss multi patch"""

    depth_patch_percent: float = 0.1
    """Percentage of patches to use for patch pearson loss"""

    depth_num_pairs: int = 1024
    """Number of pairs to use for ranking loss"""

    load_dreambooth: bool = False
    """Whether to load dreambooth model for the diffusion model"""

    sharpen_in_post: bool = False
    """Whether to sharpen the diffusion model predictions as post processing"""

    sharpen_in_post_factor: float = 1.0
    """Factor to sharpen the diffusion model predictions as post processing"""

    # NEW PARAMETERS FOR SCHEDULING:
    
    lambda_rgb_final: float = 1000.0
    """Target multiplier for RGB loss."""
    
    lambda_depth_final: float = 0.0
    """Target multiplier for depth loss."""
    
    rgb_start_step: int = 2000
    """Step to start decay RGB loss from lambda_rgb to lambda_rgb_final"""
    
    depth_start_step: int = 5000
    """Step to start decay depth loss from lambda_depth to lambda_depth_final"""
    
    rgb_lock_step: int = 13000
    """Step to stop decay RGB loss from lambda_rgb to lambda_rgb_final"""
    
    depth_lock_step: int = 13000
    """Step to stop decay depth loss from lambda_depth to lambda_depth_final"""
    
    # NEW PARAMETERS FOR RESTORATION:
    
    start_kernel_ratio: float = 0.10
    """Step ratio to start learning the deblur kernel"""

    start_diff_ratio: float = 0.25
    """Step ratio to start the diffusion distillation"""

    controlnet_tile_scale: float = 0.5
    """Conditioning scale for ControlNet Tile (lower to prevent ghosting)"""

    controlnet_depth_scale: float = 1.0
    """Conditioning scale for ControlNet Depth"""
    
    ip_adapter_scale: float = 0.5
    """Conditioning scale for Ip Adapter"""
    
    # IGNORE:

    inference_only: bool = False
    """Whether to only use the model for inference"""


class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianSplatting(Model):
    config: GaussianSplattingModelConfig
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        device,
        load_iteration: int = -1,
        orientation_transform: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(config, scene_box, num_train_data)

        self.config = config

        # if not self.config.inference_only:
        #     self.guidance.to(self.device)

        self.gaussian_model.to(self.device)
        self.gaussian_model.xyz_gradient_accum.to(self.device)
        self.gaussian_model.denom.to(self.device)

        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        self.opacity_modifier = ViewerSlider(name="Opacity Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.scale_modifier = ViewerSlider(name="Scale Slider", default_value=1.0, min_value=0.0, max_value=1.0)
        
        if self.config.deblur_enabled:
            # num_train_data is available from the super().__init__
            self.blur_kernels = nn.Parameter(
                torch.zeros((num_train_data, 1, self.config.deblur_kernel_size, self.config.deblur_kernel_size))
            )
            # Initialize as an identity kernel (sharp)
            center = self.config.deblur_kernel_size // 2
            # self.blur_kernels.data[:, 0, center, center] = 1.0
            with torch.no_grad():
                self.blur_kernels.fill_(0.0)
                self.blur_kernels[:, :, center, center] = 1.0
        

    def setup_diffusion(self):
        from realmdreamer.guidance.sd_controlnet_guidance import StableDiffusionControlNetGuidance
        self.guidance = StableDiffusionControlNetGuidance(device="cuda")
        self.diffusion_setup = True

    def populate_modules(self):
        super().populate_modules()

        # load gaussian model
        self.gaussian_model = self.config.gaussian_model.setup()

        self.gaussian_model.load_pcd(
            os.path.join(self.config.pcd_path),
            device="cuda",
            use_sigmoid=self.config.use_sigmoid,
        )
        
        self.gaussian_model.xyz_gradient_accum = torch.zeros((self.gaussian_model.get_xyz.shape[0], 1), device="cuda")
        self.gaussian_model.denom = torch.zeros((self.gaussian_model.get_xyz.shape[0], 1), device="cuda")

        # Set up losses
        # self.rgb_loss = MSELoss()
        self.rgb_loss = L1Loss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        self.lpips = lpips.LPIPS(net="vgg").to("cuda")
        
        self.ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")

        # Diffusion Guidance Setup (Only if needed)
        self.diffusion_setup = False
        if self.config.is_distillation:
            self.setup_diffusion()

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:

        viewpoint_camera = ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
        )

        render = render_results["render"]
        depth = render_results["depth"]
        alpha = render_results["alpha"]

        rgb = torch.permute(torch.clamp(render, max=1.0), (1, 2, 0))
        depth = torch.permute(depth, (1, 2, 0))
        return {
            "rgb": rgb,
            "depth": depth,
        }

    def render(
        self,
        viewpoint_camera,
        pc,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussian_model.get_xyz,
                dtype=self.gaussian_model.get_xyz.dtype,
                requires_grad=True,
                device=pc.get_xyz.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(self.device)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        # print(pc.get_xyz.device, pc.get_features.device, viewpoint_camera.camera_center.device, self.device)

        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        if self.config.average_colors:
            shs_view = (
                pc.get_averaged_features(num_neighbours=2).transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            )

        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        if self.config.use_sigmoid:
            colors_precomp = sh2rgb
            colors_precomp = wide_sigmoid(colors_precomp)

        # print(means3D.device, means3D.device, colors_precomp.device, opacity.device, scales.device, rotations.device)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        visibility_filter = radii > 0

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["xyz"] = [self.gaussian_model._xyz]
        param_groups["f_dc"] = [self.gaussian_model._features_dc]
        param_groups["f_rest"] = [self.gaussian_model._features_rest]
        param_groups["opacity"] = [self.gaussian_model._opacity]
        param_groups["scaling"] = [self.gaussian_model._scaling]
        param_groups["rotation"] = [self.gaussian_model._rotation]
            
        if self.config.deblur_enabled:
            param_groups["deblur_kernels"] = [self.blur_kernels]

        return param_groups

    def named_parameters(self, recurse=False):
        params = {}

        params["xyz"] = self.gaussian_model._xyz
        params["f_dc"] = self.gaussian_model._features_dc
        params["f_rest"] = self.gaussian_model._features_rest
        params["opacity"] = self.gaussian_model._opacity
        params["scaling"] = self.gaussian_model._scaling
        params["rotation"] = self.gaussian_model._rotation

        # Convert dict to iterator
        params = params.items()

        return params

    def forward_single(self, viewpoint_camera) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Run forward for a single camera pose
        """

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=self.device)

        # random background
        random_background = torch.tensor(np.random.uniform(0, 1, (3,)), dtype=torch.float32, device=self.device)

        start = time.time()
        out = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            # bg_color=background
            bg_color=random_background,
        )
        end = time.time()

        # print(out['render'].shape, end - start, "IMAGE")

        image = out["render"]
        depth = out["depth"]
        alpha = out["alpha"]
        viewspace_points = out["viewspace_points"]
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]

        return {
            "rgb": image,
            "depth": depth,
            "alpha": alpha,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    @torch.cuda.amp.autocast(False)
    def forward(self, cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a camera pose
        Args:
            - c2w: camera poses
        """

        viewpoint_cameras = ns2gs_camera(cameras, device=self.device)

        # assert not isinstance(viewpoint_cameras, list), "Cameras are a list is a list"
        if isinstance(viewpoint_cameras, list):

            outputs = {
                "rgb": [],
                "depth": [],
                "alpha": [],
                "viewspace_points": [],
                "visibility_filter": [],
                "radii": [],
            }

            for viewpoint_camera in viewpoint_cameras:
                out = self.forward_single(viewpoint_camera)
                for key in out.keys():
                    outputs[key].append(out[key])

            for key in outputs.keys():
                if key != "viewspace_points":
                    outputs[key] = torch.stack(outputs[key], dim=0)
        else:

            outputs = self.forward_single(viewpoint_cameras)

            for key in outputs.keys():
                if key == "viewspace_points":
                    outputs[key] = [outputs[key]]
                else:
                    outputs[key] = outputs[key].unsqueeze(0)

        return outputs

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}

        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        return metrics_dict
    
    def get_rgb_loss(self, image, gt):
        """Compute RGB loss on the full image, ignoring masks."""

        # 1. Permute to B C H W (Standard format for most PyTorch/Kornia losses)
        img_bchw = image.permute(0, 3, 1, 2)
        gt_bchw = gt.permute(0, 3, 1, 2)

        # 2. L1 Loss (Global)
        l1_loss = F.l1_loss(img_bchw, gt_bchw, reduction="mean")
        
        # 3. SSIM Loss (Global)
        # k_losses.ssim_loss returns (1 - SSIM), so it is a minimization objective
        ssim_loss = k_losses.ssim_loss(img_bchw, gt_bchw, window_size=11, reduction="mean")
        
        lambda_gs_l1 = 0.8

        # 4. Standard 3DGS weighted combination
        total_loss = lambda_gs_l1 * l1_loss + (1-lambda_gs_l1) * ssim_loss

        return total_loss

    def get_loss_dict(self, outputs, batch, prompt, step, step_ratio, max_num_iterations = 15000):

        if not self.diffusion_setup:
            self.setup_diffusion()
            self.diffusion_setup = True

        loss_dict = {}
        misc = {}

        image = batch["image"].to("cuda")
        batch["depth_image"] = batch["depth_image"].float().to("cuda")
        
        # --- 0. DYNAMIC SCHEDULING LOGIC ---
        # RGB Lambda Schedule
        if step < self.config.rgb_start_step:
            current_lambda_rgb = self.config.lambda_rgb
        elif step < self.config.rgb_lock_step:
            # Linear Interpolation: lambda_rgb -> lambda_rgb_final
            ratio = (step - self.config.rgb_start_step) / (self.config.rgb_lock_step - self.config.rgb_start_step)
            current_lambda_rgb = self.config.lambda_rgb + ratio * (self.config.lambda_rgb_final - self.config.lambda_rgb)
        else:
            current_lambda_rgb = self.config.lambda_rgb_final

        # Depth Lambda Schedule
        if step < self.config.depth_start_step:
            current_lambda_depth = self.config.lambda_depth
        elif step < self.config.depth_lock_step:
            # Linear Interpolation: lambda_depth -> lambda_depth_final
            ratio = (step - self.config.depth_start_step) / (self.config.depth_lock_step - self.config.depth_start_step)
            current_lambda_depth = self.config.lambda_depth + ratio * (self.config.lambda_depth_final - self.config.lambda_depth)
        else:
            current_lambda_depth = self.config.lambda_depth_final

        # --- 1. THE POSITION LOCK (Triggered exactly at lock step) ---
        # if step ==self.config.depth_lock_step:
        #     print(f"\n[Step {step}] >>> GEOMETRY LOCK ACTIVATED <<<")
        #     print(f"Zeroing Depth Loss and reducing Positional LR for high-frequency refinement.")
        #     for group in self.gaussian_model.optimizer.param_groups:
        #         if group["name"] == "means":
        #             group["lr"] *= 0.1  # Drop LR by 10x to stop geometry from drifting
            
        # 1. SHARP RENDERS
        sharp_rendered_rgb_bhwc = outputs["rgb"].permute(0, 2, 3, 1).clone()
        sharp_rendered_rgb_bchw = outputs["rgb"].clone()
        misc["sharp_render"] = sharp_rendered_rgb_bhwc
        
        # 2. BLUR KERNEL
        if self.config.deblur_enabled and step_ratio > self.config.start_kernel_ratio:  # Only start blurring after the scene has basic structure
            # 1. Get the image index and the corresponding kernel
            img_idx = batch["image_idx"] 
            # Use the full batch of kernels
            raw_kernels = self.blur_kernels[img_idx] # Shape: [B, 1, 15, 15]

            # 2. Normalize the kernels (CRITICAL for color/brightness stability)
            # We use abs to ensure positive weights and add epsilon to prevent division by zero
            curr_kernels = torch.abs(raw_kernels) 
            curr_kernels = curr_kernels / (curr_kernels.sum(dim=(2, 3), keepdim=True) + 1e-8)
            
            # 3. Prepare kernel for RGB depthwise convolution
            # Instead of .repeat on the whole batch, we need to repeat the channel dim
            # Shape change: [B, 1, 15, 15] -> [B*3, 1, 15, 15] to match groups=3 in a batch
            B = sharp_rendered_rgb_bchw.shape[0]
            kernel_rgb = curr_kernels.repeat_interleave(3, dim=0) 
            
            # 4. Apply blur to the RENDERED image
            p = self.config.deblur_kernel_size // 2
            
            blurred_rendered_rgb_bchw = F.conv2d(sharp_rendered_rgb_bchw, weight=kernel_rgb, padding=p, groups=3 * B)
            rendered_rgb_bhwc = blurred_rendered_rgb_bchw.permute(0, 2, 3, 1)
            
            misc["blurred_render"] = rendered_rgb_bhwc
            misc["kernel"] = curr_kernels
            
             # 5. Improved Regularization
            # Sparsity (L1) encourages a clean, sharp kernel (delta function)
            # TV loss encourages the kernel to be locally smooth (no salt-and-pepper noise)
            loss_dict["loss_kernel_reg"] = torch.mean(torch.abs(raw_kernels)) * 0.00001
            
            # Center-weighting (Total Variation) to keep the kernel focused
            diff_h = torch.abs(curr_kernels[:, :, 1:, :] - curr_kernels[:, :, :-1, :]).mean()
            diff_w = torch.abs(curr_kernels[:, :, :, 1:] - curr_kernels[:, :, :, :-1]).mean()
            loss_dict["loss_kernel_tv"] = (diff_h + diff_w) * 0.001
        else:
            rendered_rgb_bhwc = sharp_rendered_rgb_bhwc
            misc["blurred_render"] = sharp_rendered_rgb_bhwc
        
        for key in ["rgb", "depth"]:
            outputs[key] = outputs[key].permute(0, 2, 3, 1)

        # --- 2. THE GEOMETRIC ANCHOR (Direct Depth Loss) ---
        rendered_depth_bhwc = outputs["depth"]
        # Match dimensions if needed
        if rendered_depth_bhwc.shape[1:3] != batch["depth_image"].shape[1:3]:
            outputs["depth"] = F.interpolate(outputs["depth"].permute(0,3,1,2), size=batch["depth_image"].shape[1:3], mode="bilinear", align_corners=False).permute(0,2,3,1)
            rendered_depth_bhwc = outputs["depth"]
            
        gt_depth_meters = batch["depth_image"].to(self.device)
        outputs["target_depth_rescaled"] = gt_depth_meters
        
        valid_depth_mask = batch["depth_image"] > 0
        # We need at least a handful of valid pixels to compute a meaningful correlation
        if valid_depth_mask.sum() > 10 and current_lambda_depth > 0:
            # Extract only the valid depth pixels into 1D tensors
            rend_d = rendered_depth_bhwc[valid_depth_mask]
            gt_d = gt_depth_meters[valid_depth_mask]
            
            # Center the data around their respective means (this removes the shift)
            rend_centered = rend_d - rend_d.mean()
            gt_centered = gt_d - gt_d.mean()
            
            # Compute Covariance and Standard Deviations (Scale Invariance)
            cov = (rend_centered * gt_centered).sum()
            std_rend = torch.sqrt((rend_centered ** 2).sum() + 1e-8)
            std_gt = torch.sqrt((gt_centered ** 2).sum() + 1e-8)
            
            # Pearson Correlation Coefficient (r is between -1 and 1)
            pearson_corr = cov / (std_rend * std_gt)
            
            # Loss is minimized (0.0) when correlation is perfect (1.0)
            loss_dict["loss_depth"] = current_lambda_depth * (1.0 - pearson_corr)
        else:
            loss_dict["loss_depth"] = torch.tensor(0.0, device=self.device)

        
        # --- 3. THE PHYSICAL ANCHOR (Blurred RGB vs GT RGB) ---
        loss_dict["loss_rgb"] = current_lambda_rgb  * self.get_rgb_loss(rendered_rgb_bhwc, image)
        
        
        
        # --- NEW: DYNAMIC SCHEDULING (Activates strictly AFTER start_diff_ratio) ---
         # Calculate transition progress (0.0 means not started, 1.0 means handoff complete)
        # Using a 2,000 step window out of 30,000 total steps = ratio of ~0.0667
        transition_ratio = 5000.0 / max_num_iterations

        if step_ratio >= self.config.start_diff_ratio:
            # Calculate how far we are into the diffusion phase (0.0 to 1.0)
            diff_progress = min(1.0, (step_ratio - self.config.start_diff_ratio) / transition_ratio)
        else:
            diff_progress = 0.0
            
        # 2. Linear warmup for Distillation losses (0.0 -> Max)
        current_lambda_mse = self.config.lambda_one_step #* diff_progress
        current_lambda_lpips = self.config.lambda_one_step_perceptual #* diff_progress
        
    
        
        # --- 4. THE DISTILLATION ANCHOR (Diffusion) ---
        # if self.training and step_ratio > self.config.start_diff_ratio:
        #     # Pass scales down to guidance
        #     self.guidance.cfg.controlnet_conditioning_scale =[
        #         self.config.controlnet_tile_scale, 
        #         self.config.controlnet_depth_scale
        #     ]
        #     self.guidance.cfg.ip_adapter_scale = self.config.ip_adapter_scale
            
        #     # Pass SHARP render, GT RGB, and GT Depth to ControlNet
        #     pseudo_gt_sharp_bchw = self.guidance.multi_step(
        #         rgb=sharp_rendered_rgb_bhwc,
        #         scan_rgb=image,
        #         scan_depth=batch["depth_image"].permute(0, 3, 1, 2).to(self.device),
        #         prompt=prompt,
        #         current_step_ratio=step_ratio,
        #     )
            
        #     # Ensure FP32 for loss computation
        #     pseudo_gt_sharp_bchw = pseudo_gt_sharp_bchw.float()
        #     misc["pseudo_gt"] = pseudo_gt_sharp_bchw
            
        #     # Pull the SHARP 3DGS render toward the Pseudo-GT
        #     loss_dict["loss_distill_mse"] = current_lambda_mse * F.mse_loss(sharp_rendered_rgb_bchw, pseudo_gt_sharp_bchw)
        #     loss_dict["loss_distill_lpips"] = current_lambda_lpips * self.lpips(sharp_rendered_rgb_bchw * 2 - 1, pseudo_gt_sharp_bchw * 2 - 1).mean()

        # Opaqueness Loss
        clamped_opacity = torch.clamp(self.gaussian_model.get_opacity, min=1e-5, max=1.0 - 1e-5)
        loss_dict["loss_opaque"] = self.config.lambda_opaque * torch.mean(clamped_opacity * (1.0 - clamped_opacity))
        
        # Clean NaNs
        for key in loss_dict.keys():
            loss_dict[key] = torch.nan_to_num(loss_dict[key])

        return loss_dict, misc

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Create a new state_dict with only desired submodules
        filtered_state_dict = super().state_dict(destination, prefix, keep_vars)
        for name, param in self.named_parameters():
            if "guidance" not in name:
                filtered_state_dict[prefix + name] = param

        return filtered_state_dict

    def load_state_dict(self, model_dict, strict=True):
        # Reinitialize the shape of all the gaussians
        self.gaussian_model.init_random(num_points=model_dict["gaussian_model._xyz"].shape[0])
        super().load_state_dict(model_dict, strict)

    def get_stats(self):

        stats = {}

        stats["num_points"] = self.gaussian_model.get_xyz.shape[0]

        # for each field, log min, max, mean, grad_min, grad_max
        stats["xyz/min"] = torch.min(self.gaussian_model.get_xyz)
        stats["xyz/max"] = torch.max(self.gaussian_model.get_xyz)
        stats["xyz/median"] = torch.median(self.gaussian_model.get_xyz)
        stats["xyz/mean"] = torch.mean(self.gaussian_model.get_xyz)

        stats["xyz/grad_min"] = torch.min(self.gaussian_model.xyz_gradient_accum)
        stats["xyz/grad_max"] = torch.max(self.gaussian_model.xyz_gradient_accum)
        stats["xyz/grad_median"] = torch.median(self.gaussian_model.xyz_gradient_accum)
        stats["xyz/grad_mean"] = torch.mean(self.gaussian_model.xyz_gradient_accum)

        stats["features/min"] = torch.min(self.gaussian_model.get_features)
        stats["features/max"] = torch.max(self.gaussian_model.get_features)
        stats["features/median"] = torch.median(self.gaussian_model.get_features)
        stats["features/mean"] = torch.mean(self.gaussian_model.get_features)

        stats["opacity/min"] = torch.min(self.gaussian_model.get_opacity)
        stats["opacity/max"] = torch.max(self.gaussian_model.get_opacity)
        stats["opacity/median"] = torch.median(self.gaussian_model.get_opacity)
        stats["opacity/mean"] = torch.mean(self.gaussian_model.get_opacity)
        stats["opacity/low_opacity_count"] = torch.sum(self.gaussian_model.get_opacity < 0.1)

        stats["scaling/min"] = torch.min(self.gaussian_model.get_scaling)
        stats["scaling/max"] = torch.max(self.gaussian_model.get_scaling)
        stats["scaling/median"] = torch.median(self.gaussian_model.get_scaling)
        stats["scaling/mean"] = torch.mean(self.gaussian_model.get_scaling)

        stats["rotation/min"] = torch.min(self.gaussian_model.get_rotation)
        stats["rotation/max"] = torch.max(self.gaussian_model.get_rotation)

        return stats

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
        pass
