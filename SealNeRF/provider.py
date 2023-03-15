import os
import numpy as np
import torch
from tqdm import tqdm

import torch.multiprocessing as mp
from nerf.utils import get_rays
from nerf.provider import NeRFDataset, rand_poses
from .seal_utils import SealMapper
import dearpygui.dearpygui as dpg


class SealDataset(NeRFDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__(opt, device, type, downscale, n_test)
        self.proxy_flag = False
        self.depths = None

    def proxy_dataset(self, model, n_batch: int = 1):
        depths = []
        images = []
        for i in tqdm(range(len(self.poses)), desc=f'Proxying {self.type} data'):
            index = [i]
            poses = self.poses[index].to(self.device)  # [B, 4, 4]

            # copied from SealNeRF/trainer.py. always be true
            image_shape = self.images[index].shape

            error_map = None if self.error_map is None else self.error_map[index]

            rays = get_rays(poses, self.intrinsics, self.H, self.W,
                            -1, error_map, self.opt.patch_size)

            rays_o = rays['rays_o']  # [B, N, 3]
            rays_d = rays['rays_d']  # [B, N, 3]
            proxied_images = []
            proxied_depths = []

            with torch.no_grad():
                total_batches = rays_o.shape[1]
                batch_size = total_batches // n_batch
                if (total_batches % n_batch):
                    n_batch += 1
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in range(n_batch):
                        # refresh frame to prevent gui from being no response
                        if 'DISPLAY' in os.environ and dpg.is_dearpygui_running():
                            dpg.render_dearpygui_frame()
                        # dt_gamma_bak = self.opt.dt_gamma
                        # self.opt.dt_gamma = 1 / 256
                        current_teacher_outputs = model.render(
                            rays_o[:, i*batch_size:(i+1)*batch_size, :], rays_d[:, i*batch_size:(i+1)*batch_size, :], staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        # self.opt.dt_gamma = dt_gamma_bak
                        proxied_images.append(current_teacher_outputs['image'])
                        proxied_depths.append(current_teacher_outputs['depth'])
                proxied_images = torch.nan_to_num(
                    torch.concat(proxied_images, 1), nan=0.)
                proxied_depths = torch.nan_to_num(
                    torch.concat(proxied_depths, 1), nan=0.)

            proxied_images = proxied_images.view(*image_shape[:-1], -1)
            proxied_depths = proxied_depths.view(*image_shape[:-1], -1)

            images.append(proxied_images[0].detach())
            depths.append(proxied_depths[0].detach())

        torch.cuda.empty_cache()
        self.images = torch.stack(images, dim=0)
        self.depths = torch.stack(depths, dim=0)
        self.proxy_flag = True

    def collate(self, index):

        B = len(index)  # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            # only in training, assert num_rays > 0
            s = np.sqrt(self.H * self.W / self.num_rays)
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
            }

        poses = self.poses[index].to(self.device)  # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]

        rays = get_rays(poses, self.intrinsics, self.H, self.W,
                        self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'skip_proxy': self.proxy_flag,
            'data_index': torch.tensor(index),
            'pixel_index': rays['inds'] if 'inds' in rays else None
        }

        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(
                    B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            results['images'] = images

        if self.depths is not None:
            depths = self.depths[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = depths.shape[-1]
                depths = torch.gather(depths.view(
                    B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            results['depths'] = depths

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        return results

    def dataloader(self):
        loader = super().dataloader()
        loader.extra_info = {
            'H': self.H,
            'W': self.W,
            'provider': self,
        }
        return loader


# generate custom poses dataset
class SealRandomDataset(NeRFDataset):
    def __init__(self, opt, device, seal_mapper: SealMapper, type='train', downscale=1, n_test=10):
        super().__init__(opt, device, type, downscale, n_test)
        self.look_at = seal_mapper.map_data['pose_center'].to(device)
        self.radius = seal_mapper.map_data['pose_radius'].to(device)
        self.image_shape = self.images[0].shape
        if type == 'val':
            self.poses = self.poses[:10]

    def collate(self, index):

        B = len(index)  # a list of length 1

        poses = rand_poses(
            B, self.device, look_at=self.look_at, radius=self.radius)

        # sample a low-resolution but full image for CLIP
        # only in training, assert num_rays > 0
        s = np.nan_to_num(np.sqrt(self.H * self.W / self.num_rays),
                          nan=1.) if self.num_rays > 0 else 1.
        rH, rW = int(self.H / s), int(self.W / s)
        rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

        data = {
            'H': rH,
            'W': rW,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],

        }
        if self.type != 'train':
            data['images_shape'] = [B, *self.image_shape]

        return data
