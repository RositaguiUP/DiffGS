import os
import torch
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F

# --- NAFNet Minimal Architecture ---
# (Included here to keep the script self-contained)
from utils.v3dc_io import read_v3dc_iter


# class LayerNormFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, bias, eps):
#         ctx.save_for_backward(x, weight, bias)
#         ctx.eps = eps
#         mu = x.mean(-1, keepdim=True)
#         var = (x - mu).pow(2).mean(-1, keepdim=True)
#         res = (x - mu) / torch.sqrt(var + eps)
#         res = res * weight + bias
#         return res

# class LayerNorm2d(nn.Module):
#     def __init__(self, channels, eps=1e-6):
#         super().__init__()
#         self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
#         self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
#         self.eps = eps
#     def forward(self, x):
#         return LayerNormFunction.apply(x.permute(0, 2, 3, 1), self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

# class NAFBlock(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         dw_channel = c * 2
#         self.blk = nn.Sequential(
#             LayerNorm2d(c),
#             nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True),
#             nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True),
#             nn.Sequential(nn.Conv2d(dw_channel, c, 1, padding=0, stride=1, groups=1, bias=True)),
#             LayerNorm2d(c),
#             nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True),
#             nn.Conv2d(dw_channel, c, 1, padding=0, stride=1, groups=1, bias=True),
#         )
#     def forward(self, x):
#         return x + self.blk(x)

# Note: This is a simplified placeholder for the full NAFNet class. 
# In a real scenario, you would import the full NAFNet class from the NAFNet repo.
# For this script to work, ensure you have the 'models' folder from the NAFNet repo 
# or use the pre-built BasicSR wrapper.

# --- V3DC Processing Logic ---

def process_v3dc_deblur(v3dc_path, output_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model (Assumes you have the NAFNet definition imported)
    # For this example, we assume you've used: from basicsr.models.archs.nafnet_arch import NAFNet
    # from basicsr.models.archs.nafnet_arch import NAFNet
    from utils.nafnet_arch import NAFNet
    model = NAFNet(
        img_channel=3, 
        width=64, 
        middle_blk_num=1, 
        enc_blk_nums=[1, 1, 1, 28],  # Changed from [2, 2, 4, 8]
        dec_blk_nums=[1, 1, 1, 1]    # Changed from [2, 2, 2, 2]
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Pick the correct key from the checkpoint
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint

    # Load into model
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully with strict=True")
    except RuntimeError as e:
        print("\n[!] Architecture Mismatch detected. Attempting loose load...")
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False (Warning: Some layers may not be deblurring correctly)")

    model.to(device).eval()
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Iterate through V3DC frames
    print(f"Starting deblurring: {v3dc_path}")
    
    with torch.no_grad():
        for view in read_v3dc_iter(v3dc_path, read_rgb=True, read_depth=False):
            img_rgb = view['img']
            if img_rgb is None: continue

            # Pre-process: HWC to CHW, Normalize to [0, 1]
            # inp = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.
            inp = torch.from_numpy(img_rgb.copy()).float().permute(2, 0, 1) / 255.
            inp = inp.unsqueeze(0).to(device)

            # Pad to be divisible by 8 (required by most UNet architectures like NAFNet)
            h, w = inp.shape[2], inp.shape[3]
            hp = ((h + 7) // 8) * 8
            wp = ((w + 7) // 8) * 8
            padding = (0, wp - w, 0, hp - h)
            inp = F.pad(inp, padding, mode='reflect')

            # Inference
            output = model(inp)

            # Unpad and Post-process
            output = output[:, :, :h, :w]
            output = output.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
            output = (output * 255.0).round().astype(np.uint8)

            # Save Frame (Convert RGB back to BGR for OpenCV)
            out_name = f"{view['name']}.png"
            save_path = os.path.join(output_dir, out_name)
            cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            
            print(f"Processed {out_name}", end='\r')

if __name__ == "__main__":
    # Update these paths
    V3DC_FILE = "/shared/3du_data/data/2025/10/6V/6VSV7_695/process/floor_0/video.refine.v3dc"
    OUTPUT_FOLDER = "/home/rosita/tests/rendering/data/deblurred"
    CHECKPOINT = "utils/NAFNet-GoPro-width64.pth"

    process_v3dc_deblur(V3DC_FILE, OUTPUT_FOLDER, CHECKPOINT)