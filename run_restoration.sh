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
--max-num-iterations 30000 \
--logging.steps_per_log 100 \
--pipeline.datamanager.train-num-images-to-sample-from 1 \
--pipeline.datamanager.camera-optimizer.mode off \
--pipeline.densification-interval 500 \
--pipeline.density_end_iter 25000 \
--pipeline.model.guidance 'controlnet_tile' \
--pipeline.model.deblur_enabled True \
--pipeline.model.deblur_kernel_size 15 \
--pipeline.model.lambda_rgb 50.0 \
--pipeline.model.lambda_depth 10.0 \
--pipeline.model.depth_guidance False \
--pipeline.model.load_depth_guidance False \
--pipeline.model.lambda_depth_sds 0.0 \
--pipeline.model.lambda_sds 0.0 \
--pipeline.model.lambda_one_step 5.0 \
--pipeline.model.lambda_one_step_perceptual 100.0 \
--pipeline.model.max_step_percent 0.5 \
--pipeline.model.min_step_percent 0.1 \
--pipeline.model.anneal True \
--optimizers.xyz.optimizer.lr 0.005 \
--optimizers.opacity.optimizer.lr 0.05 \
--optimizers.scaling.optimizer.lr 0.005 \
--pipeline.model.pcd_path "${scene_folder_path}/pointcloud.ply" "

echo $command
eval $command