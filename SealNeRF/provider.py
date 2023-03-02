import numpy as np
import torch

from nerf.utils import get_rays
from nerf.provider import NeRFDataset, rand_poses
from .seal_utils import SealMapper


class SealDataset(NeRFDataset):
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
            'data_index': index,
            'pixel_index': rays['inds'].to('cpu') if 'inds' in rays else None
        }

        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(
                    B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            results['images'] = images

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        return results

    def dataloader(self):
        loader = super().dataloader()
        loader.extra_info = {
            'H': self.H,
            'W': self.W
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
