import os, io, csv, math, random
from importlib.metadata import files
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from einops import rearrange
import cv2
import warnings


class LargeScaleEgoTactileVideos(Dataset):
    def __init__(self, root_path, txt_path, width, height, n_sample_frames, sample_frame_rate,
                 app=None, handler_ante=None, face_helper=None):
        self.root_path = root_path
        self.txt_path = txt_path
        # Use paths relative to this file (v2p_dataset.py)
        _base_dir = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
        _v2p_dir = osp.join(_base_dir, "V2P_data")
        self.prototype_left_path = osp.join(_v2p_dir, "prototype_left.png")
        self.prototype_right_path = osp.join(_v2p_dir, "prototype_right.png")
        self.mask_left_path = osp.join(_v2p_dir, "mask_left.png")
        self.mask_right_path = osp.join(_v2p_dir, "mask_right.png")
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        
        self.video_files = self._read_txt_file_images()

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            video_files = []
            for line in lines:
                video_file = line.strip()
                video_files.append(video_file)
        return video_files

    def __len__(self):
        return len(self.video_files)
    
    def filter_files_by_string(self, directory, search_string):
        """
        Filters files in a directory that contain the given search string in their names.
        """
        try:
            files = os.listdir(directory)
            filtered_files = [file for file in files if search_string in file]
            return filtered_files
        except Exception as e:
            print(f"Error accessing directory {directory}: {e}")
            return []

    def preprocess_images_original(self, image_path, mode='RGB'):
        """
        Preprocess an image by resizing and normalizing it.
        Args:
        image_path (str): Path to the image file.
        mode (str): Mode to open the image ('RGB' for color, 'L' for grayscale).
        Returns:
        torch.Tensor: Preprocessed image tensor.
        """
        try:
            pil_image = Image.open(image_path).convert(mode)
            pil_image = pil_image.resize((self.width, self.height))
            pil_image = torch.from_numpy(np.array(pil_image)).float()
            pil_image = pil_image / 127.5 - 1
            return pil_image
        except Exception as e:
            print(f"Fail loading the image: {image_path}")
            if mode == 'RGB':
                return torch.zeros((self.height, self.width, 3))
            elif mode == 'L':
                return torch.zeros((self.height, self.width))

    def preprocess_images(self, image_path, mode='RGB'):
        """
        Preprocess an image by resizing and normalizing it.
        Args:
        image_path (str): Path to the image file.
        mode (str): 'RGB' for color image; 'L' for mask.
        Returns:
        torch.Tensor:
        - RGB: [H, W, 3], float in [-1, 1]
        - L  : [H, W, 3], float in {-1, 0, 1} (conditioning-friendly)
        """
        try:
            pil_image = Image.open(image_path).convert(mode)

            if mode == 'RGB':
                pil_image = pil_image.resize((self.width, self.height))
                img = torch.from_numpy(np.array(pil_image)).float()  # [H,W,3]
                img = img / 127.5 - 1.0
                return img
            elif mode == 'L':
                pil_image = pil_image.resize((self.width, self.height), resample=Image.NEAREST)
                mask = torch.from_numpy(np.array(pil_image)).float()  # [H,W], values ideally {0,1,2}
                mask = torch.clamp(mask, 0.0, 2.0)
                mask = mask - 1.0  # [H,W]
                mask3 = mask.unsqueeze(-1).repeat(1, 1, 3)
                return mask3
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'RGB' or 'L'.")
        except Exception as e:
            print(f"Fail loading the image: {image_path} | err={e}")
            if mode == 'RGB':
                return torch.zeros((self.height, self.width, 3))
            elif mode == 'L':
                return torch.zeros((self.height, self.width, 3))
    
    def __getitem__(self, idx):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        dir_root = self.video_files[idx]
        if "left" in dir_root:
            self.hand_side = "left"
        elif "right" in dir_root:
            self.hand_side = "right"
        else:
            self.hand_side = "right"
        
        tgt_pressure_path_list = self.filter_files_by_string(dir_root, 'pressure.png')
        tgt_pressure_path_list = sorted(tgt_pressure_path_list)
        rgb_frames_path_list = self.filter_files_by_string(dir_root, 'rgb.png')
        rgb_frames_path_list = sorted(rgb_frames_path_list)
        mask_frames_path_list = self.filter_files_by_string(dir_root, 'mask.png')
        mask_frames_path_list = sorted(mask_frames_path_list)
        
        video_length = len(tgt_pressure_path_list)
        # print(f"Video length: {video_length} frames in {dir_root}")

        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)
        # print(f"Clip length: {clip_length} frames")

        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int).tolist()

        tgt_pressure_frames_list = []
        rgb_frames_list = []
        mask_frames_list = []

        prototype_left = self.preprocess_images(self.prototype_left_path, mode='RGB')
        prototype_right = self.preprocess_images(self.prototype_right_path, mode='RGB')

        # print(f"Processing batch_index: {batch_index} in video {dir_root}")
        for index in batch_index:
            tgt_pressure_path = osp.join(dir_root, tgt_pressure_path_list[index])
            rgb_frames_path = osp.join(dir_root, rgb_frames_path_list[index])
            mask_frames_path = osp.join(dir_root, mask_frames_path_list[index])
            
            tgt_pressure_frames = self.preprocess_images(tgt_pressure_path, mode='RGB')
            tgt_pressure_frames_list.append(tgt_pressure_frames)
            rgb_frames = self.preprocess_images(rgb_frames_path, mode='RGB')
            rgb_frames_list.append(rgb_frames)
            mask_frames = self.preprocess_images(mask_frames_path, mode='L')
            mask_frames_list.append(mask_frames)

        tgt_pressure_frames_list = torch.stack(tgt_pressure_frames_list, dim=0)
        rgb_frames_list = torch.stack(rgb_frames_list, dim=0)
        mask_frames_list = torch.stack(mask_frames_list, dim=0)
        tgt_pressure_frames_list = rearrange(tgt_pressure_frames_list, "f h w c -> f c h w")
        rgb_frames_list = rearrange(rgb_frames_list, "f h w c -> f c h w")
        mask_frames_list = rearrange(mask_frames_list, "f h w c -> f c h w")
        
        if self.hand_side == "left":
            prototype = prototype_left
            mask_left = Image.open(self.mask_left_path)
            pressure_mask = mask_left
        elif self.hand_side == "right":
            prototype = prototype_right
            mask_right = Image.open(self.mask_right_path)
            pressure_mask = mask_right
        
        prototype = rearrange(prototype, "h w c -> c h w")
        
        pressure_mask = pressure_mask.convert('L')
        pressure_mask = pressure_mask.resize((self.width, self.height))
        pressure_mask = torch.from_numpy(np.array(pressure_mask)).float()
        pressure_mask = pressure_mask / 255
        pressure_mask = pressure_mask.unsqueeze(0).repeat(tgt_pressure_frames_list.size(0), 1, 1)
        pressure_mask = rearrange(pressure_mask, "f h w -> f 1 h w")

        text_prompt_path = dir_root + ".txt"
        try:
            with open(text_prompt_path, "r", encoding="utf-8") as f:
                text_prompt = f.read().strip()
        except Exception as e:
            print(f"Fail loading the text prompt: {text_prompt_path}")
            text_prompt = ""

        sample = dict(
            tgt_pressure = tgt_pressure_frames_list,
            rgb_frames=rgb_frames_list,
            mask_frames=mask_frames_list,
            prototype=prototype,
            pressure_mask=pressure_mask,
            text_prompt = text_prompt,
        )
                
        return sample



