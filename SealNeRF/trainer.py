import sys
import os
from typing import Union
import tensorboardX
import torch
import numpy as np
import tqdm
import time
import json5
import trimesh
from nerf.utils import Trainer as NGPTrainer
from tensoRF.utils import Trainer as TensoRFTrainer
from scipy.spatial.transform import Rotation
from .types import BackBoneTypes, CharacterTypes


def get_trainer(backbone: BackBoneTypes, character: CharacterTypes):
    """
    Get trainer class of `backbone` and `character` defined in `./types.py`
    """
    if character is CharacterTypes.Student:
        trainer_cls = trainer_constructor(backbone_refs[backbone], backbone)
    elif character is CharacterTypes.Teacher:
        trainer_cls = backbone_refs[backbone]
    else:
        raise NotImplementedError(f'trainer character {character.name}')
    trainer_cls._character = character
    return trainer_cls


def trainer_constructor(base, backbone: BackBoneTypes):
    """
    Construct trainer class with dynamically selected base class
    """
    Trainer = type(f'Trainer_{backbone.name}', (base,), {
        '__init__': init,
        'init_pretraining': init_pretraining,
        'train': train,
        'train_step': train_step,
        'eval_step': eval_step,
        'test_step': test_step,
        'pretrain_one_epoch': pretrain_one_epoch,
        'pretrain_part': pretrain_part,
        'pretrain_step': pretrain_step,
        'freeze_mlp': freeze_mlp,
        'set_lr': set_lr,
        'proxy_truth': proxy_truth,
        'train_gui': train_gui,
        '_backbone': backbone
    })
    Trainer._self = Trainer
    return Trainer


def get_character_constructor(character: CharacterTypes):
    def func(self):
        return character
    return func


backbone_refs = {
    BackBoneTypes.NGP: NGPTrainer,
    BackBoneTypes.TensoRF: TensoRFTrainer
}

trainer_types = Union[NGPTrainer, TensoRFTrainer]


def init(self: trainer_types, name, opt, student_model, teacher_model, proxy_train=True, proxy_test=False, proxy_eval=False, cache_gt=False, criterion=None, optimizer=None, ema_decay=None, lr_scheduler=None, metrics=..., local_rank=0, world_size=1, device=None, mute=False, fp16=False, eval_interval=1, eval_count=None, max_keep_ckpt=2, workspace='workspace', best_mode='min', use_loss_as_metric=True, report_metric_at_train=False, use_checkpoint="latest", use_tensorboardX=True, scheduler_update_every_step=False):
    super(self._self, self).__init__(name, opt, student_model, criterion=criterion, optimizer=optimizer, ema_decay=ema_decay, lr_scheduler=lr_scheduler, metrics=metrics, local_rank=local_rank, world_size=world_size, device=device, mute=mute, fp16=fp16, eval_interval=eval_interval, eval_count=eval_count,
                                     max_keep_ckpt=max_keep_ckpt, workspace=workspace, best_mode=best_mode, use_loss_as_metric=use_loss_as_metric, report_metric_at_train=report_metric_at_train, use_checkpoint=use_checkpoint, use_tensorboardX=use_tensorboardX, scheduler_update_every_step=scheduler_update_every_step)
    # use teacher trainer instead of teacher model directly
    # to make sure it's properly initialized, e.g. device
    self.teacher_model = teacher_model

    # flags indicating the proxy behavior of different stages
    self.proxy_train = proxy_train
    self.proxy_eval = proxy_eval
    self.proxy_test = proxy_test
    self.is_pretraining = False
    self.has_proxied = False

    self.cache_gt = cache_gt


def init_pretraining(self: trainer_types, epochs=0, batch_size=4096, lr=0.07,
                     local_point_step=0.001, local_angle_step=45,
                     surrounding_point_step=0.01, surrounding_angle_step=45, surrounding_bounds_extend=0.2,
                     global_point_step=0.05, global_angle_step=45):
    """
    call this until seal_mapper is initialized
    """
    # pretrain epochs before the real training starts
    self.pretraining_epochs = epochs
    self.pretraining_batch_size = batch_size
    self.pretraining_lr = lr
    if self.pretraining_epochs > 0:
        # simply use L1 to compute pretraining loss
        self.pretraining_criterion = torch.nn.L1Loss().to(self.device)
        # sample points and dirs from seal mapper
        self.pretraining_data = {}

        # prepare local data and gt
        if local_point_step > 0:
            local_bounds = self.teacher_model.seal_mapper.map_data['force_fill_bound']
            local_points, local_dirs = sample_points(
                local_bounds, local_point_step, local_angle_step)
            local_points = local_points.to(
                self.device, torch.float32)
            local_dirs = local_dirs.to(
                self.device, torch.float32)

            # map sampled points
            mapped_points, mapped_dirs, mapped_mask = self.teacher_model.seal_mapper.map_to_origin(local_points, torch.zeros_like(
                local_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))
            # if we enable map source, all points inside fill bound should be kept
            if 'map_source' in self.teacher_model.seal_mapper.map_data:
                mapped_mask[:] = True
            # filter sampled points. only store masked ones
            local_points = local_points[mapped_mask]
            N_local_points = local_points.shape[0]
            # prepare sampled dirs so we won't need to do randomly sampling in the tringing time
            local_dirs = local_dirs[torch.randint(
                local_dirs.shape[0], (N_local_points,), device=self.device)]

            # infer gt sigma & color from teacher model and store them
            mapped_points = mapped_points[mapped_mask]
            mapped_dirs = mapped_dirs[mapped_mask]

            if hasattr(self.teacher_model, 'secondary_teacher_model'):
                gt_sigma, gt_color = self.teacher_model.secondary_teacher_model(
                    mapped_points, mapped_dirs)
            else:
                gt_sigma, gt_color = self.teacher_model(
                    mapped_points, mapped_dirs)

            # map gt color
            gt_color = self.teacher_model.seal_mapper.map_color(
                mapped_points, mapped_dirs, gt_color)

            # prepare pretraining steps to avoid cuda oom
            local_steps = list(
                range(0, N_local_points, self.pretraining_batch_size))
            if local_steps[-1] != N_local_points:
                local_steps.append(N_local_points)

            self.pretraining_data['local'] = {
                'points': local_points,
                'dirs': local_dirs,
                'sigma':  gt_sigma.detach(),
                'color': gt_color.detach(),
                'steps': local_steps
            }
            self.is_pretraining = True

        # prepare surrounding data and gt
        if surrounding_point_step > 0:
            # (B, 2, 3) or (2, 3)
            surrounding_bounds: torch.Tensor = self.teacher_model.seal_mapper.map_data[
                'force_fill_bound']
            if surrounding_bounds.ndim == 2:
                surrounding_bounds[0] -= surrounding_bounds_extend
                surrounding_bounds[0] = torch.max(
                    surrounding_bounds[0], self.model.aabb_train[:3])
                surrounding_bounds[1] += surrounding_bounds_extend
                surrounding_bounds[1] = torch.min(
                    surrounding_bounds[1], self.model.aabb_train[3:])
            else:
                surrounding_bounds[:, 0] -= surrounding_bounds_extend
                surrounding_bounds[:, 0] = torch.max(
                    surrounding_bounds[:, 0], self.model.aabb_train[:3])
                surrounding_bounds[:, 1] += surrounding_bounds_extend
                surrounding_bounds[:, 1] = torch.min(
                    surrounding_bounds[:, 1], self.model.aabb_train[3:])
            surrounding_points, surrounding_dirs = sample_points(
                surrounding_bounds, surrounding_point_step, surrounding_angle_step)
            surrounding_points = surrounding_points.to(
                self.device, torch.float32)
            surrounding_dirs = surrounding_dirs.to(
                self.device, torch.float32)

            # map sampled points
            _, _, mapped_mask = self.teacher_model.seal_mapper.map_to_origin(surrounding_points, torch.zeros_like(
                surrounding_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))
            # filter sampled points. only store unmasked ones
            surrounding_points = surrounding_points[~mapped_mask]
            N_surrounding_points = surrounding_points.shape[0]
            # prepare sampled dirs so we won't need to do randomly sampling in the tringing time
            surrounding_dirs = surrounding_dirs[torch.randint(
                surrounding_dirs.shape[0], (N_surrounding_points,), device=self.device)]

            gt_sigma, gt_color = self.teacher_model(
                surrounding_points, surrounding_dirs)

            # prepare pretraining steps to avoid cuda oom
            surrounding_steps = list(
                range(0, N_surrounding_points, self.pretraining_batch_size))
            if surrounding_steps[-1] != N_surrounding_points:
                surrounding_steps.append(N_surrounding_points)

            self.pretraining_data['surrounding'] = {
                'points': surrounding_points,
                'dirs': surrounding_dirs,
                'sigma':  gt_sigma.detach(),
                'color': gt_color.detach(),
                'steps': surrounding_steps
            }

        # prepare global data and gt
        if global_point_step > 0:
            global_bounds = self.model.aabb_train.view(2, 3)
            global_points, global_dirs = sample_points(
                global_bounds, global_point_step, global_angle_step)
            global_points = global_points.to(
                self.device, torch.float32)
            global_dirs = global_dirs.to(
                self.device, torch.float32)

            _, _, mapped_mask = self.teacher_model.seal_mapper.map_to_origin(global_points, torch.zeros_like(
                global_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))

            # keep non-edited points
            global_points = global_points[~mapped_mask]
            N_global_points = global_points.shape[0]
            global_dirs = global_dirs[torch.randint(
                global_dirs.shape[0], (N_global_points,), device=self.device)]

            gt_sigma, gt_color = self.teacher_model(
                global_points, global_dirs)

            # prepare pretraining steps to avoid cuda oom
            global_steps = list(
                range(0, N_global_points, self.pretraining_batch_size))
            if global_steps[-1] != N_global_points:
                global_steps.append(N_global_points)

            self.pretraining_data['global'] = {
                'points': global_points,
                'dirs': global_dirs,
                'sigma':  gt_sigma.detach(),
                'color': gt_color.detach(),
                'steps': global_steps
            }

        visualize_dir = os.path.join(self.workspace, 'pretrain_vis')
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        for k, v in self.pretraining_data.items():
            trimesh.PointCloud(v['points'].view(-1, 3).cpu().numpy(), v['color'].view(-1, 3).cpu().numpy()).export(
                os.path.join(visualize_dir, f'{k}.ply'))


def train(self: trainer_types, train_loader, valid_loader, max_epochs):
    if self.opt.extra_epochs is not None:
        max_epochs = self.epoch + self.opt.extra_epochs

    if self.use_tensorboardX and self.local_rank == 0:
        self.writer = tensorboardX.SummaryWriter(
            os.path.join(self.workspace, "run", self.name))

    # if the model's bitfield is not hacked, do it before inferring
    if not self.teacher_model.density_bitfield_hacked:
        self.teacher_model.hack_bitfield()
    train_loader.extra_info['provider'].proxy_dataset(
        self.teacher_model, n_batch=5)
    valid_loader.extra_info['provider'].proxy_dataset(
        self.teacher_model, n_batch=5)

    # mark untrained region (i.e., not covered by any camera from the training dataset)
    if self.model.cuda_ray:
        self.model.mark_untrained_grid(
            train_loader._data.poses, train_loader._data.intrinsics)

    # get a ref to error_map
    self.error_map = train_loader._data.error_map

    if self._character is CharacterTypes.Student:
        if getattr(self.model, 'seal_mapper'):
            with open(os.path.join(self.workspace, 'seal.json'), 'w') as f:
                json5.dump(self.model.seal_mapper.config, f, quote_keys=True)
        with open(os.path.join(self.workspace, 'options.json'), 'w') as f:
            json5.dump(self.opt.__dict__, f, quote_keys=True)
        with open(os.path.join(self.workspace, 'run.sh'), 'w') as f:
            f.write(f'python {" ".join(sys.argv)}')

    # cache gt
    # ray mask
    if self.cache_gt:
        N_poses = len(train_loader)
        N_pixles = train_loader.extra_info['H'] * \
            train_loader.extra_info['W']
        self.proxy_cache_mask = torch.zeros(
            N_poses, N_pixles, dtype=torch.bool)
        self.proxy_cache_image = torch.zeros(
            N_poses, N_pixles, 3, dtype=torch.float)
        self.proxy_cache_depth = torch.zeros(
            N_poses, N_pixles, dtype=torch.float)

    first_epoch = self.epoch + 1

    time_inspector = {
        'pretraining': [],
        'pretraining_avg': 0,
        'pretraining_total': 0,
        'training': [],
        'training_avg': 0,
        'training_total': 0,
    }
    for epoch in range(self.epoch + 1, max_epochs + 1):
        self.epoch = epoch

        # is_pretraining = epoch - first_epoch < self.pretraining_epochs
        if self.is_pretraining and epoch - first_epoch >= self.pretraining_epochs:
            self.is_pretraining = False
            self.log(
                f"[INFO] Pretraining time: {sum(time_inspector['pretraining']):.4f}s")

        if self.is_pretraining:
            t = time.time()
            # skip checkpoint saving for pretraining
            self.pretrain_one_epoch()
            torch.cuda.synchronize()
            time_inspector['pretraining'].append(time.time() - t)
        else:
            self.freeze_mlp(False)
            self.set_lr(-1)
            t = time.time()
            self.train_one_epoch(train_loader)
            torch.cuda.synchronize()
            time_inspector['training'].append(time.time() - t)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

        if self.epoch % self.eval_interval == 0:
            self.evaluate_one_epoch(valid_loader)
            self.save_checkpoint(full=False, best=True)

    time_inspector['pretraining_avg'] = np.mean(time_inspector['pretraining'])
    time_inspector['pretraining_total'] = np.sum(time_inspector['pretraining'])
    time_inspector['training_avg'] = np.mean(time_inspector['training'])
    time_inspector['training_total'] = np.sum(time_inspector['training'])
    with open(os.path.join(self.workspace, 'timer.json'), 'w') as f:
        json5.dump(time_inspector, f, quote_keys=True)

    if self.use_tensorboardX and self.local_rank == 0:
        self.writer.close()


def pretrain_one_epoch(self: trainer_types, silent=False):
    """
    pretrain one epoch. set silent=True to disable logs to speed up
    """
    self.set_lr(self.pretraining_lr)

    if not self.model.density_bitfield_hacked:
        self.model.hack_bitfield()

    if not silent:
        self.log(
            f"==> Start Pre-Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

    if self.local_rank == 0 and self.report_metric_at_train:
        for metric in self.metrics:
            metric.clear()

    self.model.train()

    # freeze MLPs. this is crucial to prevent the model from being globally messed up.
    self.freeze_mlp()

    self.local_step = 0
    for part_key in self.pretraining_data.keys():
        self.pretrain_part(part_key, silent)

    if self.ema is not None:
        self.ema.update()

    if not silent:
        self.log(f"==> Finished Epoch {self.epoch}.")


def pretrain_part(self: trainer_types, source_type: str, silent: bool = False):
    source = self.pretraining_data[source_type]
    steps = source['steps']
    if not silent and self.local_rank == 0:
        pbar = tqdm.tqdm(total=len(steps) - 1,
                         bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    total_loss = 0
    for i in range(0, len(steps) - 1):
        self.local_step += 1
        self.global_step += 1

        self.optimizer.zero_grad()

        self._density_grid = self.model.density_grid

        points = source['points'][steps[i]:steps[i+1]]
        dirs = source['dirs'][steps[i]:steps[i+1]]
        # dirs = self.pretraining_dirs[torch.randint(
        #     self.pretraining_dirs.shape[0], (steps[i+1] - steps[i],), device=self.device)]
        # dirs = torch.zeros_like(
        #     points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=self.fp16):
            loss = self.pretrain_step({
                'points': points,
                'dirs': dirs,
                'indices': [steps[i], steps[i+1]],
                'source_type': source_type
            })

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # if not silent and self.scheduler_update_every_step:
        #     self.lr_scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val

        if not silent and self.local_rank == 0:
            if self.use_tensorboardX:
                self.writer.add_scalar(
                    "pretrain/loss", loss_val, self.global_step)
                self.writer.add_scalar(
                    "pretrain/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

            if self.scheduler_update_every_step:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                pbar.set_description(
                    f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(self.pretraining_batch_size)

    if not silent and self.local_rank == 0:
        pbar.close()


def pretrain_step(self: trainer_types, data):
    """
    use both sigma and color to quickly reconstruct modified space
    """
    # pred_sigma = self.model.density(data['points'])['sigma']
    source = self.pretraining_data[data['source_type']]
    pred_sigma, pred_color = self.model(data['points'], data['dirs'])
    sigma_loss = self.pretraining_criterion(
        pred_sigma, source['sigma'][data['indices'][0]:data['indices'][1]])
    color_loss = self.pretraining_criterion(
        pred_color, source['color'][data['indices'][0]:data['indices'][1]])
    # hardcoded weight. not really necessary as it is just a pretraining.
    loss = color_loss * 1 + sigma_loss
    return loss


def freeze_mlp(self: trainer_types, freeze: bool = True):
    """
    freeze all MLPs or unfreeze them by passing `freeze=False`
    """
    if self._backbone is BackBoneTypes.TensoRF:
        # freeze_module(self.model.color_net, freeze)
        # freeze_module(self.model.sigma_mat, freeze)
        # freeze_module(self.model.sigma_vec, freeze)
        # freeze_module(self.model.color_mat, freeze)
        # freeze_module(self.model.color_vec, freeze)
        # freeze_module(self.model.basis_mat, freeze)
        return
    elif self._backbone is BackBoneTypes.NGP:
        freeze_module(self.model.sigma_net, freeze)
        freeze_module(self.model.color_net, freeze)
        if hasattr(self.model, 'bg_net') and self.model.bg_net is not None:
            freeze_module(self.model.bg_net, freeze)


def set_lr(self: trainer_types, lr: float):
    """
    manually set learning rate to speedup pretraining. restore the original lr by passing `lr=-1`
    """
    if lr < 0:
        if not hasattr(self, '_cached_lr') or self._cached_lr is None:
            return
        lr = self._cached_lr
        self._cached_lr = None
    else:
        self._cached_lr = self.optimizer.param_groups[0]['lr']
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr


def proxy_truth(self: trainer_types, data, all_ray: bool = True, use_cache: bool = False, n_batch: int = 1):
    """
    proxy the ground truth RGB from teacher model
    """
    # already proxied the dataset
    if 'skip_proxy' in data and data['skip_proxy']:
        return
    # avoid OOM
    torch.cuda.empty_cache()
    # if the model's bitfield is not hacked, do it before inferring
    if not self.teacher_model.density_bitfield_hacked:
        self.teacher_model.hack_bitfield()

    # if we want a full image (B, H, W, C) or just rays (B, N, C)
    is_full = False
    if 'images' in data:
        images = data['images']  # [B, N, 3/4]
        is_full = images.ndim == 4
        image_shape = images.shape
    elif 'images_shape' in data:
        image_shape = data['images_shape']
        is_full = len(image_shape) == 4

    # if use_cache, the training can be slower. might be useful with very small dataset
    use_cache = use_cache and 'pixel_index' in data and not is_full
    is_skip_computing = False

    if use_cache:
        compute_mask = ~self.proxy_cache_mask[data['data_index'],
                                              data['pixel_index']]
        is_skip_computing = not compute_mask.any()

    rays_o = data['rays_o']  # [B, N, 3]
    rays_d = data['rays_d']  # [B, N, 3]

    if use_cache:
        rays_o = rays_o[compute_mask]
        rays_d = rays_d[compute_mask]

    if not is_skip_computing:
        teacher_outputs = {
            'image': [],
            'depth': []
        }
        with torch.no_grad():
            total_batches = rays_o.shape[1]
            batch_size = total_batches // n_batch
            if (total_batches % n_batch):
                n_batch += 1
            for i in range(n_batch):
                current_teacher_outputs = self.teacher_model.render(
                    rays_o[:, i*batch_size:(i+1)*batch_size, :], rays_d[:, i*batch_size:(i+1)*batch_size, :], staged=True, bg_color=None, perturb=False, force_all_rays=all_ray, **vars(self.opt))
                teacher_outputs['image'].append(
                    current_teacher_outputs['image'])
                teacher_outputs['depth'].append(
                    current_teacher_outputs['depth'])
            teacher_outputs['image'] = torch.concat(
                teacher_outputs['image'], 1)
            teacher_outputs['depth'] = torch.concat(
                teacher_outputs['depth'], 1)

    if use_cache:
        if not is_skip_computing:
            self.proxy_cache_image[data['data_index'], data['pixel_index']
                                   [compute_mask]] = torch.nan_to_num(teacher_outputs['image'], nan=0.).detach().cpu()
            self.proxy_cache_depth[data['data_index'], data['pixel_index']
                                   [compute_mask]] = torch.nan_to_num(teacher_outputs['depth'], nan=0.).detach().cpu()
        data['images'] = self.proxy_cache_image[data['data_index'],
                                                data['pixel_index']].to(self.device)
        data['depths'] = self.proxy_cache_depth[data['data_index'],
                                                data['pixel_index']].to(self.device)
    else:
        data['images'] = torch.nan_to_num(teacher_outputs['image'], nan=0.)
        data['depths'] = torch.nan_to_num(teacher_outputs['depth'], nan=0.)
    # reshape if it is a full image
    if is_full:
        data['images'] = data['images'].view(*image_shape[:-1], -1)
        data['depths'] = data['depths'].view(*image_shape[:-1], -1)


def train_step(self: trainer_types, data):
    # if self.teacher_model.density_bitfield_hacked:
    #     self.teacher_model.restore_bitfield()
    if self.proxy_train:
        self.proxy_truth(data, use_cache=self.cache_gt)
    return super(self._self, self).train_step(data)

@torch.no_grad()
def eval_step(self: trainer_types, data):
    if self.proxy_eval:
        self.proxy_truth(data, n_batch=5)
    return super(self._self, self).eval_step(data)

@torch.no_grad()
def test_step(self: trainer_types, data, bg_color=None, perturb=False):
    if self.proxy_test:
        self.proxy_truth(data, n_batch=5)
    return super(self._self, self).test_step(data, bg_color, perturb)


def sample_points(bounds: torch.Tensor, point_step=0.005, angle_step=45):
    """
    Sample points per step inside bounds (B, 2, 3) or (2, 3)
    """
    if bounds.ndim == 2:
        bounds = bounds[None]
    sampled_points = []
    sampled_dirs = []
    for i in range(bounds.shape[0]):
        coords_min, coords_max = bounds[i]
        X, Y, Z = torch.meshgrid(torch.arange(coords_min[0], coords_max[0], step=point_step),
                                 torch.arange(
            coords_min[1], coords_max[1], step=point_step),
            torch.arange(coords_min[2], coords_max[2], step=point_step))
        sampled_points.append(torch.stack(
            [X, Y, Z], dim=-1).reshape(-1, 3))

        r_x, r_y, r_z = torch.meshgrid(torch.arange(0, 360, step=angle_step),
                                       torch.arange(0, 360, step=angle_step),
                                       torch.arange(0, 360, step=angle_step))
        eulers = torch.stack([r_x, r_y, r_z], dim=-1).reshape(-1, 3)
        sampled_dirs.append(torch.from_numpy(Rotation.from_euler('xyz', eulers.numpy(
        ), degrees=True).apply(np.array([1-1e-5, 0, 0]))))

    # trimesh.PointCloud(
    #     self.sampled_points.cpu().numpy()).export('tmp/sampled.obj')
    return torch.concat(sampled_points), torch.concat(sampled_dirs)


def freeze_module(module: Union[torch.nn.ParameterList, torch.nn.ModuleList, torch.nn.Module], freeze: bool):
    module.training = not freeze
    if isinstance(module, (torch.nn.ParameterList, torch.nn.ModuleList)):
        for i in range(len(module)):
            module[i].requires_grad_(not freeze)
    elif isinstance(module, torch.nn.Module):
        module.requires_grad_(not freeze)


def train_gui(self, train_loader, step=16, is_pretraining=False):

    # print(is_pretraining)

    self.model.train()

    # mark untrained grid
    # if self.global_step == 0:
    #     self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

    total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

    if is_pretraining:
        st = time.time()
        torch.cuda.synchronize()
        for _ in range(step):
            self.pretrain_one_epoch(True)
        ed = time.time()
        torch.cuda.synchronize()
        self.log(f"[INFO]Pretraining epoch x{step} time: {ed-st:.4f}s")
        return {
            'loss': 0.0,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    self.freeze_mlp(False)
    self.set_lr(-1)

    # if the model's bitfield is not hacked, do it before inferring
    if not self.teacher_model.density_bitfield_hacked:
        self.teacher_model.hack_bitfield()
    if not self.has_proxied:
        train_loader.extra_info['provider'].proxy_dataset(
            self.teacher_model, n_batch=1)
        self.has_proxied = True

    loader = iter(train_loader)

    for _ in range(step):

        # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
        try:
            data = next(loader)
        except StopIteration:
            loader = iter(train_loader)
            data = next(loader)

        # update grid every 16 steps
        if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        self.global_step += 1

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.fp16):
            preds, truths, loss = self.train_step(data)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler_update_every_step:
            self.lr_scheduler.step()

        total_loss += loss.detach()

    if self.ema is not None:
        self.ema.update()

    average_loss = total_loss.item() / step

    if not self.scheduler_update_every_step:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(average_loss)
        else:
            self.lr_scheduler.step()

    outputs = {
        'loss': average_loss,
        'lr': self.optimizer.param_groups[0]['lr'],
    }

    return outputs
