import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from nerf.renderer import NeRFRenderer
from .utils import custom_meshgrid
from .seal_utils import get_seal_mapper
from .color_utils import rgb2hsv_torch, hsv2rgb_torch, rgb2hsl_torch, hsl2rgb_torch


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 /
                           n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


def modify_hsv(rgb: torch.Tensor, modification: torch.Tensor):
    N = rgb.shape[0]
    hsv = rgb2hsv_torch(rgb.view(1, 3, N))
    hsv[:, 0, :] += modification[0]
    hsv[:, 1, :] += modification[1]
    hsv[:, 2, :] += modification[2]
    return hsv2rgb_torch(hsv).view(N, 3)

# TODO: fix color bugs and keep lightness
# the original color is not correct makes the converted hsl value meaningless


def modify_rgb(rgb: torch.Tensor, modification: torch.Tensor):
    N = rgb.shape[0]
    return modification.repeat(N).view(N, 3).to(rgb.device, rgb.dtype)
    # hsl = rgb2hsl_torch(rgb.view(1, 3, N))
    # hsl_modification = rgb2hsl_torch(modification.view(1, 3, 1)).view(3).to(rgb.device, rgb.dtype)
    # hsl[:, 2, :] = hsl_modification[2]
    # return hsl2rgb_torch(hsl).view(N, 3)


class SealNeRFRenderer(NeRFRenderer):
    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1, **kwargs):
        super().__init__(bound=bound, cuda_ray=cuda_ray, density_scale=density_scale,
                         min_near=min_near, density_thresh=density_thresh, bg_radius=bg_radius)
        self.seal_mapper = None
        # Setting bitfield preciously
        # force_fill_bitfield_offsets: np.ndarray = self.force_fill_grid_indices % 8
        # self.force_fill_bitfield_values = torch.rand(
        #     [self.force_fill_bitfield_indices.shape[0], 1], dtype=torch.uint8, device=self.density_bitfield.device)
        # self.force_fill_bitfield_values = 255
        self.density_bitfield_origin = None
        self.density_bitfield_hacked = False

    def init_mapper(self, seal_config_path: str):
        self.seal_mapper = get_seal_mapper(seal_config_path)
        coords_min, coords_max = torch.floor(
            ((self.seal_mapper.map_data['force_fill_bound'] + self.bound) / self.bound / 2) * self.grid_size)
        X, Y, Z = torch.meshgrid(torch.arange(coords_min[0], coords_max[0]),
                                 torch.arange(coords_min[1], coords_max[1]),
                                 torch.arange(coords_min[2], coords_max[2]))
        coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        self.force_fill_grid_indices = raymarching.morton3D(
            coords).long()  # [N]

        self.force_fill_bitfield_indices = self.force_fill_grid_indices // 8

    def update_extra_state(self, decay=0.95, S=128):
        # self.force_fill_grids()
        super().update_extra_state(decay, S)
        if self.seal_mapper is not None:
            self.hack_bitfield()

    @torch.no_grad()
    def hack_grids(self):
        self.density_grid[:,
                          self.force_fill_grid_indices] = min(self.mean_density * 1.5, self.density_thresh) + 1e-5

    @torch.no_grad()
    def hack_bitfield(self):
        if self.density_bitfield_origin is None:
            self.density_bitfield_origin = self.density_bitfield[self.force_fill_bitfield_indices]
        self.density_bitfield[self.force_fill_bitfield_indices] = 255
        self.density_bitfield_hacked = True
        # self.density_bitfield[:,
        #                       self.force_fill_bitfield_indices] = torch.bitwise_or(self.density_bitfield[:,
        #                                                                                                  self.force_fill_bitfield_indices], self.force_fill_bitfield_values)

    @torch.no_grad()
    def restore_bitfield(self):
        self.density_bitfield[self.force_fill_bitfield_indices] = self.density_bitfield_origin
        self.density_bitfield_hacked = False


class SealNeRFTeacherRenderer(SealNeRFRenderer):
    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1, **kwargs):
        super().__init__(bound=bound, cuda_ray=cuda_ray,
                         density_scale=density_scale, min_near=min_near, density_thresh=density_thresh, bg_radius=bg_radius)

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(
            rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps,
                                device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + \
                (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = rays_o.unsqueeze(-2) + \
            rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        if self.seal_mapper is not None:
            mapped_xyzs, _, _ = self.seal_mapper.map_to_origin(
                xyzs.view(-1, 3))[0].view(xyzs.shape)
        else:
            mapped_xyzs = xyzs

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(mapped_xyzs.reshape(-1, 3))

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat(
                    [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale *
                                       density_outputs['sigma'].squeeze(-1))  # [N, T]
                alphas_shifted = torch.cat(
                    [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+1]
                weights = alphas * \
                    torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 *
                              deltas[..., :-1])  # [N, T-1]
                new_z_vals = sample_pdf(
                    z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach()  # [N, t]

                # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = rays_o.unsqueeze(-2) + \
                    rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1)
                # a manual clip.
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])

                if self.seal_mapper is not None:
                    mapped_new_xyzs, _, _ = self.seal_mapper.map_to_origin(
                        new_xyzs.view(-1, 3))[0].view(new_xyzs.shape)
                else:
                    mapped_new_xyzs = new_xyzs

            # only forward new points to save computation
            new_density_outputs = self.density(mapped_new_xyzs.reshape(-1, 3))
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(
                xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat(
                    [density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(
                    tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat(
            [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale *
                               density_outputs['sigma'].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * \
            torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        if self.seal_mapper is not None:
            mapped_final_xyzs, mapped_final_dirs, mapped_mask = self.seal_mapper.map_to_origin(
                xyzs.view(-1, 3), dirs.view(-1, 3))
        else:
            mapped_final_xyzs, mapped_final_dirs = xyzs, dirs

        mask = weights > 1e-4  # hard coded
        rgbs = self.color(mapped_final_xyzs, mapped_final_dirs,
                          mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        #print(xyzs.shape, 'valid_rgb:', mask.sum().item())

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs,
                          dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(
                rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        return {
            'depth': depth,
            'image': image,
            'weights_sum': weights_sum,
        }

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(
            rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(
                rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d)  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(
                rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

            if self.seal_mapper is not None:
                mapped_xyzs, mapped_dirs, mapped_mask = self.seal_mapper.map_to_origin(
                    xyzs.view(-1, 3), dirs.view(-1, 3))
            else:
                mapped_xyzs, mapped_dirs = xyzs, dirs

            # trimesh.PointCloud(
            #     xyzs.reshape(-1, 3).detach().cpu().numpy()).export('tmp/xyzs.obj')
            # trimesh.PointCloud(
            #     mapped_xyzs.reshape(-1, 3).detach().cpu().numpy()).export('tmp/xyzs_mapped.obj')

            sigmas, rgbs = self(mapped_xyzs.view(xyzs.shape),
                                mapped_dirs.view(dirs.shape))
            # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            # sigmas = density_outputs['sigma']
            # rgbs = self.color(xyzs, dirs, **density_outputs)
            sigmas = self.density_scale * sigmas

            if self.seal_mapper is not None:
                if 'hsv' in self.seal_mapper.map_data:
                    rgbs[mapped_mask] = modify_hsv(
                        rgbs[mapped_mask], self.seal_mapper.map_data['hsv'])
                if 'rgb' in self.seal_mapper.map_data:
                    rgbs[mapped_mask] = modify_rgb(
                        rgbs[mapped_mask], self.seal_mapper.map_data['rgb'])

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # special case for CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(
                        sigmas[k], rgbs[k], deltas, rays, T_thresh)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))

                depth = torch.stack(depths, axis=0)  # [K, B, N]
                image = torch.stack(images, axis=0)  # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(
                    sigmas, rgbs, deltas, rays, T_thresh)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)

            results['weights_sum'] = weights_sum

        else:

            # allocate outputs
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(
                n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < max_steps:

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield,
                                                            self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                if self.seal_mapper is not None:
                    mapped_xyzs, mapped_dirs, mapped_mask = self.seal_mapper.map_to_origin(
                        xyzs.view(-1, 3), dirs.view(-1, 3))
                else:
                    mapped_xyzs, mapped_dirs = xyzs, dirs

                sigmas, rgbs = self(mapped_xyzs.view(
                    xyzs.shape), mapped_dirs.view(dirs.shape))
                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                sigmas = self.density_scale * sigmas

                if self.seal_mapper is not None:
                    if 'hsv' in self.seal_mapper.map_data and len(mapped_mask):
                        rgbs[mapped_mask] = modify_hsv(
                            rgbs[mapped_mask], self.seal_mapper.map_data['hsv'])
                    if 'rgb' in self.seal_mapper.map_data:
                        rgbs[mapped_mask] = modify_rgb(
                            rgbs[mapped_mask], self.seal_mapper.map_data['rgb'])
                    if 'sigma' in self.seal_mapper.map_data:
                        sigmas[mapped_mask] += self.seal_mapper.map_data
                raymarching.composite_rays(
                    n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

        results['depth'] = depth
        results['image'] = image

        return results


class SealNeRFStudentRenderder(SealNeRFRenderer):
    def __init__(self, bound=1, cuda_ray=False, density_scale=1, min_near=0.2, density_thresh=0.01, bg_radius=-1, **kwargs):
        super().__init__(bound=bound, cuda_ray=cuda_ray,
                         density_scale=density_scale, min_near=min_near, density_thresh=density_thresh, bg_radius=bg_radius)
