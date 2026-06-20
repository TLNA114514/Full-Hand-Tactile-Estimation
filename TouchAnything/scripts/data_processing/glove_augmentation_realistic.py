#!/usr/bin/env python3
"""
Realistic Glove Region Data Augmentation Tool

This script applies realistic visual transformations to glove regions detected by SAM3.
Focus on realistic augmentations that could appear in real-world scenarios:
- Skin tone variations (different ethnicities)
- Fabric patterns (stripes, dots, textures)
- Natural lighting variations
- Realistic textures

Removed unrealistic augmentations:
- Pure colors (green, red, blue)
- Inverted colors
- Grayscale (hands are rarely grayscale in real scenarios)
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


# Realistic fabric/clothing colors
FABRIC_COLORS = [
    # Common clothing colors
    (20, 20, 20),      # Black
    (40, 40, 40),      # Dark gray
    (80, 80, 80),      # Gray
    (200, 200, 200),   # Light gray
    (240, 240, 240),   # White
    (30, 30, 80),      # Navy blue
    (60, 60, 120),     # Blue
    (40, 60, 100),     # Dark blue
    (20, 40, 60),      # Deep blue
    (80, 50, 30),      # Brown
    (100, 70, 50),     # Light brown
    (60, 40, 30),      # Dark brown
    (40, 50, 60),      # Charcoal
    (100, 100, 120),   # Blue-gray
]


def load_masks(mask_file: Path) -> Dict:
    """Load masks from npz file."""
    data = np.load(mask_file)
    
    # New format: (T, N, H, W) with all frames
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


def augment_skin_tone(image: np.ndarray, mask: np.ndarray, skin_tone: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Replace glove region with realistic skin tone (preserves texture and lighting).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        skin_tone: Target skin tone in BGR, if None, randomly selected
    
    Returns:
        Augmented image with skin tone
    """
    result = image.copy()
    
    # Select random skin tone if not provided
    if skin_tone is None:
        skin_tone = SKIN_TONES[np.random.randint(0, len(SKIN_TONES))]
    
    # Extract masked region
    masked_pixels = result[mask > 0].astype(np.float32)
    
    # Convert to LAB color space for better color transfer
    masked_pixels_uint8 = masked_pixels.astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(masked_pixels_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Convert target skin tone to LAB
    target_bgr = np.array(skin_tone, dtype=np.uint8).reshape(1, 1, 3)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Preserve luminance (L channel), replace color (A, B channels)
    # This keeps the original lighting and texture
    lab[:, :, 1] = target_lab[0, 0, 1]
    lab[:, :, 2] = target_lab[0, 0, 2]
    
    # Convert back to BGR
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    
    result[mask > 0] = bgr
    return result


def augment_skin_color_solid(image: np.ndarray, mask: np.ndarray, skin_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Replace glove region with solid skin color (no texture preservation).
    Replace completely with a solid skin color without preserving texture.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        skin_color: Target skin color in BGR, if None, randomly selected
    
    Returns:
        Augmented image with solid skin color
    """
    result = image.copy()
    
    # Select random skin color if not provided
    if skin_color is None:
        skin_color = SKIN_TONES[np.random.randint(0, len(SKIN_TONES))]
    
    # Simple solid color replacement
    result[mask > 0] = skin_color
    return result


def augment_fabric_color(image: np.ndarray, mask: np.ndarray, fabric_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Replace glove region with realistic fabric color (preserves texture).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        fabric_color: Target fabric color in BGR, if None, randomly selected
    
    Returns:
        Augmented image with fabric color
    """
    result = image.copy()
    
    # Select random fabric color if not provided
    if fabric_color is None:
        fabric_color = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    
    # Extract masked region
    masked_pixels = result[mask > 0].astype(np.float32)
    
    # Convert to LAB color space
    masked_pixels_uint8 = masked_pixels.astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(masked_pixels_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Convert target color to LAB
    target_bgr = np.array(fabric_color, dtype=np.uint8).reshape(1, 1, 3)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Preserve luminance, replace color
    lab[:, :, 1] = target_lab[0, 0, 1]
    lab[:, :, 2] = target_lab[0, 0, 2]
    
    # Convert back
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    
    result[mask > 0] = bgr
    return result


def augment_horizontal_stripes(image: np.ndarray, mask: np.ndarray, stripe_width: int = None) -> np.ndarray:
    """
    Add horizontal stripes pattern to glove region.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        stripe_width: Width of stripes in pixels
    
    Returns:
        Augmented image with horizontal stripes
    """
    result = image.copy()
    h, w = mask.shape
    
    # Random stripe width if not provided
    if stripe_width is None:
        stripe_width = np.random.randint(5, 20)
    
    # Create stripe pattern
    stripes = (np.arange(h) // stripe_width) % 2
    stripes = stripes.reshape(h, 1).repeat(w, axis=1)
    
    # Select two random colors for stripes
    color1 = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    color2 = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    
    # Apply stripes only to masked region
    for i in range(3):  # BGR channels
        stripe_channel = np.where(stripes, color1[i], color2[i]).astype(np.uint8)
        result[:, :, i] = np.where(mask > 0, stripe_channel, result[:, :, i])
    
    return result


def augment_vertical_stripes(image: np.ndarray, mask: np.ndarray, stripe_width: int = None) -> np.ndarray:
    """
    Add vertical stripes pattern to glove region.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        stripe_width: Width of stripes in pixels
    
    Returns:
        Augmented image with vertical stripes
    """
    result = image.copy()
    h, w = mask.shape
    
    # Random stripe width if not provided
    if stripe_width is None:
        stripe_width = np.random.randint(5, 20)
    
    # Create stripe pattern
    stripes = (np.arange(w) // stripe_width) % 2
    stripes = stripes.reshape(1, w).repeat(h, axis=0)
    
    # Select two random colors for stripes
    color1 = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    color2 = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    
    # Apply stripes only to masked region
    for i in range(3):  # BGR channels
        stripe_channel = np.where(stripes, color1[i], color2[i]).astype(np.uint8)
        result[:, :, i] = np.where(mask > 0, stripe_channel, result[:, :, i])
    
    return result


def augment_checkerboard(image: np.ndarray, mask: np.ndarray, checker_size: int = None) -> np.ndarray:
    """
    Add checkerboard pattern to glove region.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        checker_size: Size of checker squares in pixels
    
    Returns:
        Augmented image with checkerboard pattern
    """
    result = image.copy()
    h, w = mask.shape
    
    # Random checker size if not provided
    if checker_size is None:
        checker_size = np.random.randint(10, 30)
    
    # Create checkerboard pattern
    checker = ((np.arange(h).reshape(h, 1) // checker_size) + 
               (np.arange(w).reshape(1, w) // checker_size)) % 2
    
    # Select two random colors
    color1 = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    color2 = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    
    # Apply checkerboard only to masked region
    for i in range(3):  # BGR channels
        checker_channel = np.where(checker, color1[i], color2[i]).astype(np.uint8)
        result[:, :, i] = np.where(mask > 0, checker_channel, result[:, :, i])
    
    return result


def augment_dots_pattern(image: np.ndarray, mask: np.ndarray, dot_spacing: int = None) -> np.ndarray:
    """
    Add polka dots pattern to glove region.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        dot_spacing: Spacing between dots in pixels
    
    Returns:
        Augmented image with dots pattern
    """
    result = image.copy()
    h, w = mask.shape
    
    # Random dot spacing if not provided
    if dot_spacing is None:
        dot_spacing = np.random.randint(15, 35)
    
    dot_radius = dot_spacing // 3
    
    # Base color
    base_color = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    dot_color = FABRIC_COLORS[np.random.randint(0, len(FABRIC_COLORS))]
    
    # Create base
    for i in range(3):
        result[:, :, i] = np.where(mask > 0, base_color[i], result[:, :, i])
    
    # Add dots
    y_coords = np.arange(0, h, dot_spacing)
    x_coords = np.arange(0, w, dot_spacing)
    
    for y in y_coords:
        for x in x_coords:
            # Offset every other row for better pattern
            x_offset = dot_spacing // 2 if (y // dot_spacing) % 2 else 0
            x_pos = x + x_offset
            
            if x_pos < w:
                # Draw dot
                cv2.circle(result, (x_pos, y), dot_radius, dot_color, -1)
    
    # Apply mask
    result = np.where(mask[:, :, None] > 0, result, image)
    
    return result


def augment_brightness_variation(image: np.ndarray, mask: np.ndarray, factor_range: Tuple[float, float] = (0.6, 1.4)) -> np.ndarray:
    """
    Apply realistic brightness variation (simulating lighting changes).
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        factor_range: Range of brightness factors
    
    Returns:
        Augmented image with brightness variation
    """
    result = image.copy()
    
    # Random brightness factor
    brightness_factor = np.random.uniform(factor_range[0], factor_range[1])
    
    # Apply to masked region
    masked_pixels = result[mask > 0].astype(np.float32)
    masked_pixels = np.clip(masked_pixels * brightness_factor, 0, 255).astype(np.uint8)
    result[mask > 0] = masked_pixels
    
    return result


def augment_subtle_noise(image: np.ndarray, mask: np.ndarray, noise_level: int = 15) -> np.ndarray:
    """
    Add subtle noise to simulate fabric texture.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        noise_level: Maximum noise amplitude
    
    Returns:
        Augmented image with subtle noise
    """
    result = image.copy()
    h, w = mask.shape
    
    # Generate noise
    noise = np.random.randint(-noise_level, noise_level + 1, size=(h, w, 3), dtype=np.int16)
    
    # Apply to masked region
    masked_pixels = result[mask > 0].astype(np.int16)
    masked_pixels = np.clip(masked_pixels + noise[mask > 0], 0, 255).astype(np.uint8)
    result[mask > 0] = masked_pixels
    
    return result


def augment_slight_blur(image: np.ndarray, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply slight blur to simulate motion or focus.
    
    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask
        kernel_size: Blur kernel size (odd number)
    
    Returns:
        Augmented image with slight blur
    """
    result = image.copy()
    
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Blur the entire image
    blurred = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    # Apply only to masked region
    result[mask > 0] = blurred[mask > 0]
    
    return result


# Dictionary of all augmentation functions
AUGMENTATION_FUNCTIONS = {
    # Fixed skin tones with texture (preserve texture) - keep a fixed color throughout the video
    'skin_tone_very_light': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[0]),
    'skin_tone_light': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[3]),
    'skin_tone_medium_light': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[6]),
    'skin_tone_medium': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[9]),
    'skin_tone_medium_dark': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[12]),
    'skin_tone_dark': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[15]),
    'skin_tone_very_dark': lambda img, mask: augment_skin_tone(img, mask, SKIN_TONES[18]),
    
    # Solid skin colors (solid-color replacement,without preserving texture) - replace completely with a skin color
    'skin_color_very_light': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[0]),
    'skin_color_light': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[3]),
    'skin_color_medium_light': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[6]),
    'skin_color_medium': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[9]),
    'skin_color_medium_dark': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[12]),
    'skin_color_dark': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[15]),
    'skin_color_very_dark': lambda img, mask: augment_skin_color_solid(img, mask, SKIN_TONES[18]),
    
    # Random skin tone (random per frame for training diversity)
    'skin_tone_random': augment_skin_tone,
    'skin_color_random': augment_skin_color_solid,
    
    # Fixed fabric colors - keep a fixed color throughout the video
    'fabric_black': lambda img, mask: augment_fabric_color(img, mask, FABRIC_COLORS[0]),
    'fabric_gray': lambda img, mask: augment_fabric_color(img, mask, FABRIC_COLORS[2]),
    'fabric_white': lambda img, mask: augment_fabric_color(img, mask, FABRIC_COLORS[4]),
    'fabric_navy': lambda img, mask: augment_fabric_color(img, mask, FABRIC_COLORS[5]),
    'fabric_blue': lambda img, mask: augment_fabric_color(img, mask, FABRIC_COLORS[6]),
    'fabric_brown': lambda img, mask: augment_fabric_color(img, mask, FABRIC_COLORS[9]),
    
    # Random fabric color (random per frame)
    'fabric_random': augment_fabric_color,
    
    # Patterns - fixed patterns; stripe width and colors are sampled at video start
    'horizontal_stripes': augment_horizontal_stripes,
    'vertical_stripes': augment_vertical_stripes,
    'checkerboard': augment_checkerboard,
    'dots_pattern': augment_dots_pattern,
    
    # Natural variations - may vary per frame
    'brightness_variation': augment_brightness_variation,
    'subtle_noise': augment_subtle_noise,
    'slight_blur': augment_slight_blur,
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
    all_augmented = []
    aug_names = []
    
    for aug_name in tqdm(AUGMENTATION_FUNCTIONS.keys(), desc="Generating augmentations"):
        for sample_idx in range(num_samples):
            try:
                augmented = apply_augmentation(frame, mask, aug_name)
                
                # Save individual augmentation
                aug_filename = f"{aug_idx:02d}_{aug_name}_{sample_idx}.jpg"
                aug_path = output_dir / aug_filename
                cv2.imwrite(str(aug_path), augmented)
                
                all_augmented.append(augmented)
                aug_names.append(f"{aug_name}_{sample_idx}")
                aug_idx += 1
                
            except Exception as e:
                print(f"Warning: Failed to apply {aug_name}: {e}")
    
    print(f"\n✓ Generated {len(all_augmented)} augmented images in: {output_dir}")
    
    # Create grid visualization
    create_grid_visualization(frame, all_augmented, aug_names, output_dir)


def create_grid_visualization(original: np.ndarray, augmented_images: List[np.ndarray], 
                              aug_names: List[str], output_dir: Path):
    """Create a grid visualization of all augmentations."""
    num_images = len(augmented_images) + 1  # +1 for original
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    h, w = original.shape[:2]
    grid_h = rows * h
    grid_w = cols * w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Add original
    grid[0:h, 0:w] = original
    cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add augmented images
    for idx, (aug_img, aug_name) in enumerate(zip(augmented_images, aug_names), 1):
        row = idx // cols
        col = idx % cols
        y1 = row * h
        y2 = y1 + h
        x1 = col * w
        x2 = x1 + w
        
        grid[y1:y2, x1:x2] = aug_img
        
        # Add label
        label = aug_name.replace('_', ' ')[:20]  # Truncate long names
        cv2.putText(grid, label, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    grid_path = output_dir / "augmentation_grid.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"\nAugmentation types tested: {len(AUGMENTATION_FUNCTIONS)}")
    print(f"Samples per type: {len(augmented_images) // len(AUGMENTATION_FUNCTIONS)}")
    print(f"✓ Saved grid visualization: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Realistic glove region data augmentation")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--masks', type=str, required=True, help='Path to masks.npz file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--frame', type=int, default=100, help='Frame index to visualize')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples per augmentation type')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    mask_file = Path(args.masks)
    output_dir = Path(args.output)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    visualize_augmentations(
        video_path=video_path,
        mask_file=mask_file,
        output_dir=output_dir,
        frame_idx=args.frame,
        num_samples=args.samples,
    )


if __name__ == '__main__':
    main()
