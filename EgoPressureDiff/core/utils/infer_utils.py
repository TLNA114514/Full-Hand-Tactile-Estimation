import os
from typing import List, Sequence, Union
import numpy as np
from PIL import Image

FrameT = Union[Image.Image, np.ndarray]
FramesT = Union[List[FrameT], np.ndarray]


def _to_pil_list(video_frames: FramesT) -> List[Image.Image]:
    """Convert frames container to List[PIL.Image] for processing."""
    if isinstance(video_frames, np.ndarray):
        # assume [F,H,W,3]
        return [Image.fromarray(f.astype(np.uint8)) for f in video_frames]
    # list/sequence
    pil_list = []
    for f in video_frames:
        if isinstance(f, Image.Image):
            pil_list.append(f)
        elif isinstance(f, np.ndarray):
            pil_list.append(Image.fromarray(f.astype(np.uint8)))
        else:
            raise TypeError(f"Unsupported frame type: {type(f)}")
    return pil_list


def _restore_type_like(original: FramesT, pil_frames: List[Image.Image]) -> FramesT:
    """Restore output container/type to match original video_frames."""
    if isinstance(original, np.ndarray):
        arr = np.stack([np.array(im) for im in pil_frames], axis=0)
        return arr
    # list-like: match element type
    if len(original) == 0:
        return pil_frames  # edge case
    first = original[0]
    if isinstance(first, Image.Image):
        return pil_frames
    if isinstance(first, np.ndarray):
        return [np.array(im) for im in pil_frames]
    # fallback
    return pil_frames


def concat_external_rgb_pressure_with_generated(
    video_frames: FramesT,
    batch_index: Sequence[int],
    validation_dir: str,
) -> FramesT:
    """
    If validation_dir contains '{idx:05d}_rgb.png' for every idx in batch_index:
      - if ALSO contains '{idx:05d}_pressure.png' for every idx: concat [pressure | rgb | generated] (width x3)
      - else: concat [rgb | generated] (width x2)

    Returns frames with the SAME container/type as input video_frames.
    """
    if video_frames is None:
        return video_frames

    pil_gen = _to_pil_list(video_frames)
    if len(pil_gen) == 0:
        return video_frames

    # Target size from generated frames
    w, h = pil_gen[0].size

    rgb_paths = [os.path.join(validation_dir, f"{int(i):05d}_rgb.png") for i in batch_index]
    prs_paths = [os.path.join(validation_dir, f"{int(i):05d}_pressure.png") for i in batch_index]

    has_all_rgb = all(os.path.exists(p) for p in rgb_paths)
    has_all_prs = all(os.path.exists(p) for p in prs_paths)

    # Nothing to do
    if not has_all_rgb:
        return video_frames

    use_pressure = has_all_prs

    out_pil: List[Image.Image] = []
    for t in range(len(pil_gen)):
        gen = pil_gen[t].convert("RGB")
        rgb = Image.open(rgb_paths[t]).convert("RGB").resize((w, h), resample=Image.BILINEAR)

        if use_pressure:
            prs = Image.open(prs_paths[t]).convert("RGB").resize((w, h), resample=Image.BILINEAR)
            # order: pressure, rgb, generated  （与你描述的“pressure和rgb与video_frames按照顺序”一致）
            canvas = Image.new("RGB", (w * 3, h))
            canvas.paste(prs, (0, 0))
            canvas.paste(rgb, (w, 0))
            canvas.paste(gen, (w * 2, 0))
        else:
            # order: rgb, generated
            canvas = Image.new("RGB", (w * 2, h))
            canvas.paste(rgb, (0, 0))
            canvas.paste(gen, (w, 0))

        out_pil.append(canvas)

    return _restore_type_like(video_frames, out_pil)
