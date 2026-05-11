#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide the scene name as an argument"
  exit 1
fi

scene_name=$1

if [ -f "configs/${scene_name}.yaml" ]; then
  prompt=$(cat "configs/${scene_name}.yaml" | grep "prompt" | cut -d ":" -f 2 | head -1)
else
  echo "No config file found for scene '${scene_name}' named ${scene_name}.yaml"
  exit 1
fi

ckpt_name="2026-05-07_215701"

# Appending ControlNet directives to your prompt
# prompt="${prompt}, 4k image, photorealistic, cinematic lighting, sharp, high resolution, highly detailed texture"

scene_folder_path="outputs/${scene_name}"
command="ns-train realmdreamer --data "${scene_folder_path%/}" \
--project_name "RealmDreamer_Restoration" \
--experiment_name '${scene_name}' \
--pipeline.prompt "${prompt}" \
--vis wandb \
--machine.num-devices 1 \
--save-only-latest-checkpoint True \
--steps_per_save 1000 \
--max-num-iterations 15000 \
--logging.steps_per_log 100 \
--pipeline.datamanager.train-num-images-to-sample-from 1 \
--pipeline.datamanager.camera-optimizer.mode off \
--pipeline.densification-interval 100 \
--pipeline.density_end_iter 13000 \
--pipeline.model.guidance 'controlnet_tile' \
--pipeline.model.deblur_enabled False \
--pipeline.model.deblur_kernel_size 15 \
--pipeline.model.lambda_rgb 10000.0 \
--pipeline.model.target_rgb_loss 5000.0 \
--pipeline.model.lambda_depth 5000.0 \
--pipeline.model.depth_guidance False \
--pipeline.model.load_depth_guidance False \
--pipeline.model.lambda_depth_sds 0.0 \
--pipeline.model.lambda_sds 0.0 \
--pipeline.model.lambda_one_step 5.0 \
--pipeline.model.lambda_one_step_perceptual 100.0 \
--pipeline.model.max_step_percent 0.95 \
--pipeline.model.min_step_percent 0.25 \
--pipeline.model.anneal True \
--pipeline.model.start_kernel_ratio 1.50 \
--pipeline.model.start_diff_ratio 1.50 \
--pipeline.model.controlnet_tile_scale 0.95 \
--pipeline.model.controlnet_depth_scale 0.95 \
--pipeline.model.ip_adapter_scale 0.5 \
--optimizers.xyz.optimizer.lr 0.0008 \
--optimizers.f-dc.optimizer.lr 0.001 \
--optimizers.opacity.optimizer.lr 0.05 \
--optimizers.scaling.optimizer.lr 0.007 \
--optimizers.rotation.optimizer.lr 0.01 \
--optimizers.deblur_kernels.optimizer.lr 0.01 \
--pipeline.model.pcd_path "${scene_folder_path}/pointcloud.ply" "

echo $command
eval $command