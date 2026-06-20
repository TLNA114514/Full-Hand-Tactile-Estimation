"""
Glove Region Data Augmentation Tool

This script applies various visual transformations to glove regions detected by SAM3.
The goal is to make the model learn hand pose-tactile relationships rather than glove appearance.

Augmentation methods:
1. Skin tone replacement - Replace glove with realistic skin colors
2. Random color replacement - Replace with random solid colors
3. Color jitter - Adjust brightness, contrast, saturation, hue
4. Texture overlay - Add different textures to glove region
5. Gaussian blur - Blur the glove region
6. Random erase - Randomly erase parts of glove
7. Grayscale - Convert glove region to grayscale
8. Invert colors - Invert colors in glove region
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import argparse
from tqdm import tqdm


# Realistic skin tone colors in BGR format
# Based on actual human skin tones across different ethnicities
SKIN_TONES = [
    # Light skin tones (European, East Asian light)
    (200, 215, 235),  # Very light pink
    (185, 205, 230),  # Light pink
    (170, 195, 225),  # Fair
    (155, 185, 220),  # Light beige
    
    # Medium light skin tones (Mediterranean, Asian)
    (140, 170, 210),  # Beige
    (125, 155, 195),  # Light tan
    (115, 145, 185),  # Tan
    (105, 135, 175),  # Medium beige
    
    # Medium skin tones (Latin, Middle Eastern, South Asian)
    (95, 125, 165),   # Medium tan
    (85, 115, 155),   # Olive
    (75, 105, 145),   # Medium brown
    (65, 95, 135),    # Brown
    
    # Medium dark skin tones (South Asian, African American)
    (60, 85, 120),    # Dark tan
    (55, 75, 110),    # Dark brown
    (50, 70, 100),    # Deep brown
    
    # Dark skin tones (African, South Indian)
    (45, 60, 90),     # Very dark brown
    (40, 55, 80),     # Dark chocolate
    (35, 50, 70),     # Deep chocolate
    (30, 45, 60),     # Very deep
    (25, 40, 55),     # Ebony
]


def load_masks(mask_file: Path) -> Dict:
    """Load masks from npz file."""
    data = np.load(mask_file)
    
    # New format: (T, N, H, W) with all frames
    # Old format: had 'frame_indices' array
    masks = data['masks']
    num_frames = masks.shape[0]
    
    # Generate frame indices (0, 1, 2, ..., T-1)
    frame_indices = np.arange(num_frames)
    
    return {
        'masks': masks,
        'obj_ids': data['obj_ids'],
        'frame_indices': frame_indices,
        'valid_frames': data.get('valid_frames', np.ones(num_frames, dtype=bool)),
    }


def get_frame_mask(masks_data: Dict, frame_idx: int) -> Optional[np.ndarray]:
    """
    Get combined mask for a specific frame.
    
    Returns:
        Combined binary mask (H, W) or None if frame not found
    """
    # Check if frame index is valid
    num_frames = len(masks_data['frame_indices'])
    if frame_idx < 0 or frame_idx >= num_frames:
        return None
    
    # Check if frame has valid detection
    valid_frames = masks_data.get('valid_frames')
    if valid_frames is not None and not valid_frames[frame_idx]:
        return None
    
    masks = masks_data['masks'][frame_idx]  # (N, H, W)
    obj_ids = masks_data['obj_ids'][frame_idx]  # (N,)
    
    # Combine all valid object masks
    valid_masks = masks[obj_ids >= 0]
    if len(valid_masks) == 0:
        return None
    
    combined_mask = np.any(valid_masks > 0, axis=0).astype(np.uint8)
    return combined_mask


def augment_skin_tone_transfer(image: np.ndarray, mask: np.ndarray, skin_tone: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Replace glove region with skin tone using color transfer (preserves texture and shape).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        skin_tone: BGR color tuple, if None use random skin tone
    
    Returns:
        Augmented image with skin tone (texture preserved)
    """
    result = image.copy()
    
    if skin_tone is None:
        skin_tone = SKIN_TONES[np.random.randint(0, len(SKIN_TONES))]
    
    # Add some variation to make it more natural
    variation = np.random.randint(-10, 10, size=3)
    skin_tone = tuple(np.clip(np.array(skin_tone) + variation, 0, 255).astype(int))
    
    # Color transfer: preserve brightness and texture, only change hue/saturation
    masked_pixels = result[mask > 0].reshape(-1, 1, 3)
    
    # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
    lab = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Convert target skin tone to LAB
    target_bgr = np.array(skin_tone, dtype=np.uint8).reshape(1, 1, 3)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Keep original L (lightness/brightness), use target A and B (color)
    lab[:, :, 1] = target_lab[0, 0, 1]  # A channel (green-red)
    lab[:, :, 2] = target_lab[0, 0, 2]  # B channel (blue-yellow)
    
    # Convert back to BGR
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    
    result[mask > 0] = bgr
    return result


def augment_skin_tone_solid(image: np.ndarray, mask: np.ndarray, skin_tone: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Replace glove region with solid skin tone color (no texture preservation).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        skin_tone: BGR color tuple, if None use random skin tone
    
    Returns:
        Augmented image with solid skin tone
    """
    result = image.copy()
    
    if skin_tone is None:
        skin_tone = SKIN_TONES[np.random.randint(0, len(SKIN_TONES))]
    
    # Add some variation to make it more natural
    variation = np.random.randint(-10, 10, size=3)
    skin_tone = tuple(np.clip(np.array(skin_tone) + variation, 0, 255).astype(int))
    
    # Simple solid color replacement
    result[mask > 0] = skin_tone
    return result


def augment_random_color_transfer(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace glove region with random color using color transfer (preserves texture).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
    
    Returns:
        Augmented image with random color (texture preserved)
    """
    result = image.copy()
    
    # Generate random color
    random_color = tuple(np.random.randint(0, 256, size=3).tolist())
    
    # Use color transfer technique
    masked_pixels = result[mask > 0].reshape(-1, 1, 3)
    
    # Convert to LAB
    lab = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Convert target color to LAB
    target_bgr = np.array(random_color, dtype=np.uint8).reshape(1, 1, 3)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Keep original L, use target A and B
    lab[:, :, 1] = target_lab[0, 0, 1]
    lab[:, :, 2] = target_lab[0, 0, 2]
    
    # Convert back
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    
    result[mask > 0] = bgr
    return result


def augment_random_color_solid(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace glove region with solid random color (no texture preservation).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
    
    Returns:
        Augmented image with solid random color
    """
    result = image.copy()
    
    # Generate random color
    random_color = tuple(np.random.randint(0, 256, size=3).tolist())
    
    # Simple solid color replacement
    result[mask > 0] = random_color
    return result


def augment_color_jitter(image: np.ndarray, mask: np.ndarray, 
                         brightness: float = 0.4, 
                         contrast: float = 0.4,
                         saturation: float = 0.4,
                         hue: float = 0.2) -> np.ndarray:
    """Apply color jitter to glove region."""
    result = image.copy()
    
    # Extract masked region
    masked_pixels = result[mask > 0].astype(np.float32)
    
    # Apply brightness
    brightness_factor = np.random.uniform(1 - brightness, 1 + brightness)
    masked_pixels = np.clip(masked_pixels * brightness_factor, 0, 255)
    
    # Apply contrast
    mean = masked_pixels.mean()
    contrast_factor = np.random.uniform(1 - contrast, 1 + contrast)
    masked_pixels = np.clip((masked_pixels - mean) * contrast_factor + mean, 0, 255)
    
    # Convert to HSV for saturation and hue adjustment
    masked_pixels_uint8 = masked_pixels.astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(masked_pixels_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply saturation
    saturation_factor = np.random.uniform(1 - saturation, 1 + saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    # Apply hue shift
    hue_shift = np.random.uniform(-hue * 180, hue * 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Convert back to BGR
    hsv_uint8 = hsv.astype(np.uint8)
    masked_pixels_bgr = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    
    # Put back
    result[mask > 0] = masked_pixels_bgr
    
    return result


def augment_texture_overlay(image: np.ndarray, mask: np.ndarray, texture_type: str = 'noise') -> np.ndarray:
    """Add texture overlay to glove region."""
    result = image.copy()
    h, w = mask.shape
    
    if texture_type == 'noise':
        # Random noise texture
        noise = np.random.randint(0, 50, size=(h, w, 3), dtype=np.uint8)
        result[mask > 0] = np.clip(result[mask > 0].astype(int) + noise[mask > 0], 0, 255).astype(np.uint8)
    
    elif texture_type == 'gradient':
        # Gradient texture
        gradient = np.linspace(0, 100, w).reshape(1, w, 1).repeat(h, axis=0).repeat(3, axis=2).astype(np.uint8)
        result[mask > 0] = np.clip(result[mask > 0].astype(int) + gradient[mask > 0], 0, 255).astype(np.uint8)
    
    elif texture_type == 'checkerboard':
        # Checkerboard pattern
        checker_size = 20
        checker = np.indices((h, w)).sum(axis=0) // checker_size % 2
        checker = (checker * 50).astype(np.uint8)
        checker = np.stack([checker] * 3, axis=-1)
        result[mask > 0] = np.clip(result[mask > 0].astype(int) + checker[mask > 0], 0, 255).astype(np.uint8)
    
    return result


def augment_gaussian_blur(image: np.ndarray, mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply Gaussian blur to glove region."""
    result = image.copy()
    
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Blur entire image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Apply only to masked region
    result[mask > 0] = blurred[mask > 0]
    
    return result


def augment_random_erase(image: np.ndarray, mask: np.ndarray, erase_ratio: float = 0.3) -> np.ndarray:
    """Randomly erase parts of glove region."""
    result = image.copy()
    
    # Create random erase mask
    erase_mask = np.random.random(mask.shape) < erase_ratio
    combined_mask = (mask > 0) & erase_mask
    
    # Fill with random color or mean color
    fill_color = tuple(np.random.randint(0, 256, size=3).tolist())
    result[combined_mask] = fill_color
    
    return result


def augment_grayscale(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert glove region to grayscale."""
    result = image.copy()
    
    # Convert masked region to grayscale
    masked_pixels = result[mask > 0]
    gray = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).reshape(-1, 3)
    
    result[mask > 0] = gray_bgr
    
    return result


def augment_invert_colors(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Invert colors in glove region."""
    result = image.copy()
    result[mask > 0] = 255 - result[mask > 0]
    return result


def augment_hsv_shift(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply random HSV shift to glove region."""
    result = image.copy()
    
    # Extract masked region
    masked_pixels = result[mask > 0].reshape(-1, 1, 3)
    
    # Convert to HSV
    hsv = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Random shifts
    h_shift = np.random.uniform(-30, 30)
    s_shift = np.random.uniform(-50, 50)
    v_shift = np.random.uniform(-50, 50)
    
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)
    
    # Convert back
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    result[mask > 0] = bgr
    
    return result


# Dictionary of all augmentation functions
AUGMENTATION_FUNCTIONS = {
    # Skin tone - color transfer (preserves texture) - RECOMMENDED
    'skin_tone_transfer': augment_skin_tone_transfer,
    # Skin tone - solid color (no texture)
    'skin_tone_solid': augment_skin_tone_solid,
    # Random color - color transfer (preserves texture) - RECOMMENDED
    'random_color_transfer': augment_random_color_transfer,
    # Random color - solid color (no texture)
    'random_color_solid': augment_random_color_solid,
    # Other augmentations
    'color_jitter': augment_color_jitter,
    'texture_noise': lambda img, mask: augment_texture_overlay(img, mask, 'noise'),
    'texture_gradient': lambda img, mask: augment_texture_overlay(img, mask, 'gradient'),
    'texture_checkerboard': lambda img, mask: augment_texture_overlay(img, mask, 'checkerboard'),
    'gaussian_blur': augment_gaussian_blur,
    'random_erase': augment_random_erase,
    'grayscale': augment_grayscale,
    'invert_colors': augment_invert_colors,
    'hsv_shift': augment_hsv_shift,
}


def apply_augmentation(image: np.ndarray, mask: np.ndarray, aug_type: str) -> np.ndarray:
    """
    Apply specified augmentation to image.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        aug_type: Augmentation type name
    
    Returns:
        Augmented image
    """
    if aug_type not in AUGMENTATION_FUNCTIONS:
        raise ValueError(f"Unknown augmentation type: {aug_type}. Available: {list(AUGMENTATION_FUNCTIONS.keys())}")
    
    return AUGMENTATION_FUNCTIONS[aug_type](image, mask)


def visualize_augmentations(video_path: Path, mask_file: Path, output_dir: Path, 
                           frame_idx: int = 0, num_samples: int = 3):
    """
    Generate visualization of all augmentation methods on a sample frame.
    
    Args:
        video_path: Path to input video
        mask_file: Path to masks.npz file
        output_dir: Directory to save visualization results
        frame_idx: Frame index to visualize
        num_samples: Number of random samples per augmentation type
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load masks
    print(f"Loading masks from: {mask_file}")
    masks_data = load_masks(mask_file)
    
    # Load video frame
    print(f"Loading frame {frame_idx} from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from video")
    
    # Get mask for this frame
    mask = get_frame_mask(masks_data, frame_idx)
    if mask is None:
        raise ValueError(f"No mask found for frame {frame_idx}")
    
    print(f"Frame shape: {frame.shape}, Mask shape: {mask.shape}")
    print(f"Mask coverage: {(mask > 0).sum() / mask.size * 100:.2f}%")
    
    # Save original frame
    original_path = output_dir / "00_original.jpg"
    cv2.imwrite(str(original_path), frame)
    print(f"✓ Saved original: {original_path}")
    
    # Save mask visualization
    mask_vis = np.zeros_like(frame)
    mask_vis[mask > 0] = [0, 255, 0]  # Green for mask
    mask_overlay = cv2.addWeighted(frame, 0.6, mask_vis, 0.4, 0)
    mask_path = output_dir / "01_mask_overlay.jpg"
    cv2.imwrite(str(mask_path), mask_overlay)
    print(f"✓ Saved mask overlay: {mask_path}")
    
    # Apply each augmentation type
    aug_idx = 2
    for aug_name in tqdm(AUGMENTATION_FUNCTIONS.keys(), desc="Generating augmentations"):
        for sample_idx in range(num_samples):
            try:
                augmented = apply_augmentation(frame, mask, aug_name)
                
                # Save augmented result
                output_path = output_dir / f"{aug_idx:02d}_{aug_name}_sample{sample_idx+1}.jpg"
                cv2.imwrite(str(output_path), augmented)
                
                aug_idx += 1
            except Exception as e:
                print(f"Warning: Failed to apply {aug_name}: {e}")
    
    print(f"\n✓ Generated {aug_idx - 2} augmented images in: {output_dir}")
    print(f"\nAugmentation types tested: {len(AUGMENTATION_FUNCTIONS)}")
    print(f"Samples per type: {num_samples}")
    
    # Create a grid visualization
    create_grid_visualization(output_dir, output_dir / "augmentation_grid.jpg")


def create_grid_visualization(image_dir: Path, output_path: Path, max_images: int = 20):
    """Create a grid visualization of augmented images."""
    import glob
    
    # Get all images
    image_files = sorted(glob.glob(str(image_dir / "*.jpg")))[:max_images]
    
    if len(image_files) == 0:
        print("No images found for grid visualization")
        return
    
    # Load images
    images = [cv2.imread(f) for f in image_files]
    
    # Resize images to smaller size for grid
    target_size = (320, 240)
    images = [cv2.resize(img, target_size) for img in images]
    
    # Calculate grid size
    n_images = len(images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create grid
    grid = np.zeros((n_rows * target_size[1], n_cols * target_size[0], 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        y_start = row * target_size[1]
        x_start = col * target_size[0]
        grid[y_start:y_start+target_size[1], x_start:x_start+target_size[0]] = img
        
        # Add label
        label = Path(image_files[idx]).stem
        cv2.putText(grid, label, (x_start + 5, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(str(output_path), grid)
    print(f"✓ Saved grid visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Glove Region Data Augmentation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--masks", type=str, required=True, help="Path to masks.npz file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples per augmentation type")
    
    args = parser.parse_args()
    
    visualize_augmentations(
        video_path=Path(args.video),
        mask_file=Path(args.masks),
        output_dir=Path(args.output),
        frame_idx=args.frame,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
