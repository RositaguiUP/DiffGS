import os
import argparse
from pathlib import Path

import cv2
import torch
import kornia
from kornia.enhance import sharpness


def load_image_tensor(img_path):
    """
    Load image with OpenCV -> convert to RGB tensor [1, C, H, W] in range [0,1]
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0          # H,W,C
    img = img.permute(2, 0, 1).unsqueeze(0)              # 1,C,H,W
    return img


def save_image_tensor(img_tensor, save_path):
    """
    Save tensor [1,C,H,W] in range [0,1]
    """
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img)


def main(path, sharpness_factor):
    input_folder = Path(path)

    if not input_folder.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    # Create output folder with "_sharp"
    output_folder = input_folder.parent / f"{input_folder.name}_sharp"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Supported formats
    exts = [".png", ".jpg", ".jpeg", ".bmp"]

    images = [p for p in input_folder.iterdir() if p.suffix.lower() in exts]

    print(f"Found {len(images)} images")

    for img_path in images:
        print(f"Processing {img_path.name}")

        img = load_image_tensor(img_path)

        # Apply sharpness
        img_sharp = sharpness(img, factor=float(sharpness_factor))

        # Save
        save_path = output_folder / img_path.name
        save_image_tensor(img_sharp, save_path)

    print(f"Saved sharpened images to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to folder containing RGB images",
    )

    parser.add_argument(
        "--sharpness-factor",
        type=float,
        default=4.0,
        help="Sharpness factor (1.0 = original, >1 sharper)",
    )

    args = parser.parse_args()

    main(args.path, args.sharpness_factor)