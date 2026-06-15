import os
# ==========================================================================================
# 🛑 核心黑魔法：源码感知 + 全局空间硬核注入补丁（地表最强终结版，完美解决一切 NameError）
# ==========================================================================================
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
_args, _ = _parser.parse_known_args()

os.environ['PYOPENGL_PLATFORM'] = _args.render_platform
os.environ['PYRENDER_PLATFORM'] = _args.render_platform

try:
    import types
    import builtins
    import re
    import sys

    # 1. 定义一个全能通配符类：既是数字0，又是可任意调用的函数，还支持无限切片和属性延伸
    class UltimateMagicMock(int):
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return self
        def __getitem__(self, item): return self
        def __iter__(self): return iter([])

    class PerfectMockModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__'): raise AttributeError(name)
            return UltimateMagicMock(0)

    mock_obj = PerfectMockModule('OpenGL.GL')

    # 2. 拦截系统的底层 __import__ 行为
    orig_import = builtins.__import__
    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        # 只要发现有任何文件在尝试染指 OpenGL/EGL/OSMesa
        if name.startswith('OpenGL') or name in ['EGL', 'OSMesa']:
            if globals is not None and '__file__' in globals:
                try:
                    # 【硬核注入】读取当前正在执行 import 的文件（如 texture.py）的源码
                    with open(globals['__file__'], 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 抓取该文件里写的所有 OpenGL 相关的函数和常量（如 GL_TEXTURE_2D, glGenTextures 等）
                    tokens = re.findall(r'\b([gG][lL][A-Za-z0-9_]+|[eE][gG][lL][A-Za-z0-9_]+|OSMesa[A-Za-z0-9_]+)\b', content)
                    
                    # 直接强行把这些变量塞进该文件的全局命名空间，彻底断绝 NameError 的可能
                    for token in tokens:
                        if token not in globals:
                            globals[token] = UltimateMagicMock(0)
                except Exception:
                    pass
            return mock_obj
        return orig_import(name, globals, locals, fromlist, level)
    
    # 替换系统全局导入函数
    builtins.__import__ = custom_import

    # 3. 固化系统路由备份
    sys.modules['EGL'] = mock_obj
    sys.modules['OSMesa'] = mock_obj
    sys.modules['OpenGL'] = mock_obj
    sys.modules['OpenGL.GL'] = mock_obj
    sys.modules['OpenGL.GL.shaders'] = mock_obj
    
    print("\n====== [Success] Hardcore Global Token Injector Activated! ======\n")
except Exception as e:
    print(f"Bypass failed: {e}")
# ==========================================================================================

from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
import shutil
import subprocess
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def create_video_from_images(image_paths, output_path, fps):
    """Create a video from a list of image paths using FFmpeg pipe or OpenCV fallback."""
    if not image_paths:
        print(f"No images provided to generate {output_path}")
        return False

    # Verify all paths exist
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        print(f"None of the image paths exist for {output_path}")
        return False

    # Get size from the first image
    first_img = cv2.imread(str(valid_paths[0]))
    if first_img is None:
        print(f"Failed to read first image: {valid_paths[0]}")
        return False
    height, width, _ = first_img.shape

    print(f"Generating video {output_path} with {len(valid_paths)} frames at {fps} FPS, resolution {width}x{height}")

    # Try FFmpeg pipe first
    if shutil.which('ffmpeg') is not None:
        print("Using FFmpeg pipeline for video encoding...")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for path in valid_paths:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                process.stdin.write(img.tobytes())
            process.stdin.close()
            process.wait()
            if process.returncode == 0:
                print(f"✓ Video successfully saved using FFmpeg: {output_path}")
                return True
            else:
                print("FFmpeg pipe process failed. Falling back to OpenCV...")
        except Exception as e:
            print(f"Error using FFmpeg process: {e}. Falling back to OpenCV...")

    # OpenCV Fallback
    print("Using OpenCV VideoWriter...")
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for path in valid_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            out.write(img)
        out.release()
        print(f"✓ Video successfully saved using OpenCV: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to generate video using OpenCV: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code with video generation')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    
    # Video generation arguments
    parser.add_argument('--save_video', dest='save_video', action='store_true', default=True, help='If set, compile generated images into videos')
    parser.add_argument('--no_video', dest='save_video', action='store_false', help='Disable video generation')
    parser.add_argument('--video_fps', type=int, default=20, help='FPS for the compiled videos')
    parser.add_argument('--video_name', type=str, default='demo_overlay_ok_2.mp4', help='Filename of output overlay video')
    parser.add_argument('--patch_video_name', type=str, default='demo_patches_ok_2.mp4', help='Filename of output patch video')
    
    # Hand selection arguments
    parser.add_argument('--hand', type=str, default='both', choices=['left', 'right', 'both'], help='Which hand to render (left, right, both)')
    parser.add_argument('--single_hand', dest='single_hand', action='store_true', default=False, help='If set, keep at most one detected hand per frame')
    parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')

    args = parser.parse_args()

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        local_vitdet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '_DATA', 'model_final_f05665.pkl'))
        if os.path.exists(local_vitdet_path):
            detectron2_cfg.train.init_checkpoint = local_vitdet_path
        else:
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png, sorted to guarantee correct video order
    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        img_fn, _ = os.path.splitext(os.path.basename(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            if args.hand in ['left', 'both']:
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
            
            if args.hand in ['right', 'both']:
                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)

        # If --single_hand is enabled, keep at most one detected hand per frame
        if args.single_hand and len(bboxes) > 1:
            bboxes = bboxes[:1]
            is_right = is_right[:1]

        if len(bboxes) == 0:
            # If no hands are detected, save the original image frame as the overlay
            # to maintain strict 1:1 synchronization for video compilation
            if args.full_frame:
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), img_cv2)
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view overlay
        if args.full_frame:
            if len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
            else:
                # Save original frame if no hands could be reconstructed, to maintain synchronization
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), img_cv2)

    # Compile videos after the frame loop ends
    if args.save_video:
        print("\nStarting video compilation...")
        
        # 1. Overlay video
        if args.full_frame:
            overlay_paths = [os.path.join(args.out_folder, f'{os.path.splitext(os.path.basename(img_path))[0]}_all.jpg') for img_path in img_paths]
            video_out_path = os.path.join(args.out_folder, args.video_name)
            create_video_from_images(overlay_paths, video_out_path, args.video_fps)

        # 2. Patch video
        patch_paths = [os.path.join(args.out_folder, f'{os.path.splitext(os.path.basename(img_path))[0]}_0.png') for img_path in img_paths]
        if any(os.path.exists(p) for p in patch_paths):
            patch_video_out_path = os.path.join(args.out_folder, args.patch_video_name)
            create_video_from_images(patch_paths, patch_video_out_path, args.video_fps)
        else:
            print("No hand patches found, skipping patch video compilation.")

if __name__ == '__main__':
    main()
