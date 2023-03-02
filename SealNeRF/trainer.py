import os
from typing import Union
import tensorboardX
import torch
import numpy as np
import tqdm
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
    return trainer_cls


def trainer_constructor(base, backbone: BackBoneTypes):
    """
    Construct trainer class with dynamically selected base class
    """
    Trainer = type(f'Trainer_{backbone}', (base,), {
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
        '_backbone': backbone
    })
    Trainer._self = Trainer
    return Trainer


backbone_refs = {
    BackBoneTypes.NGP: NGPTrainer,
    BackBoneTypes.TensoRF: TensoRFTrainer
}

trainer_types = Union[NGPTrainer, TensoRFTrainer]


def init(self: trainer_types, name, opt, student_model, teacher_trainer, proxy_train=True, proxy_test=False, proxy_eval=False, cache_gt=False, criterion=None, optimizer=None, ema_decay=None, lr_scheduler=None, metrics=..., local_rank=0, world_size=1, device=None, mute=False, fp16=False, eval_interval=1, eval_count=None, max_keep_ckpt=2, workspace='workspace', best_mode='min', use_loss_as_metric=True, report_metric_at_train=False, use_checkpoint="latest", use_tensorboardX=True, scheduler_update_every_step=False):
    super(self._self, self).__init__(name, opt, student_model, criterion=criterion, optimizer=optimizer, ema_decay=ema_decay, lr_scheduler=lr_scheduler, metrics=metrics, local_rank=local_rank, world_size=world_size, device=device, mute=mute, fp16=fp16, eval_interval=eval_interval, eval_count=eval_count,
                                     max_keep_ckpt=max_keep_ckpt, workspace=workspace, best_mode=best_mode, use_loss_as_metric=use_loss_as_metric, report_metric_at_train=report_metric_at_train, use_checkpoint=use_checkpoint, use_tensorboardX=use_tensorboardX, scheduler_update_every_step=scheduler_update_every_step)
    # use teacher trainer instead of teacher model directly
    # to make sure it's properly initialized, e.g. device
    self.teacher_trainer = teacher_trainer

    # flags indicating the proxy behavior of different stages
    self.proxy_train = proxy_train
    self.proxy_eval = proxy_eval
    self.proxy_test = proxy_test

    self.cache_gt = cache_gt


# call this until seal_mapper is initialized
def init_pretraining(self: trainer_types, epochs=0, local_point_step=0.01, local_angle_step=45, global_point_step=0.05, global_angle_step=45, batch_size=4096, lr=0.07):
    # pretrain epochs before the real training starts
    self.pretraining_epochs = epochs
    self.pretraining_batch_size = batch_size
    self.pretraining_lr = lr
    if self.pretraining_epochs > 0:
        # simply use L1 to compute pretraining loss
        self.pretraining_criterion = torch.nn.L1Loss().to(self.device)
        # sample points and dirs from seal mapper
        self.pretraining_data = {
            'global': {},
            'local': {}
        }

        # prepare local data and gt
        local_bounds = self.teacher_trainer.model.seal_mapper.map_data['force_fill_bound']
        local_points, local_dirs = sample_points(
            local_bounds, local_point_step, local_angle_step)
        local_points = local_points.to(
            self.device, torch.float32)
        local_dirs = local_dirs.to(
            self.device, torch.float32)

        # map sampled points
        mapped_points, mapped_dirs, mapped_mask = self.teacher_trainer.model.seal_mapper.map_to_origin(local_points, torch.zeros_like(
            local_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))
        # filter sampled points. only store masked ones
        local_points = local_points[mapped_mask]
        N_local_points = local_points.shape[0]
        # prepare sampled dirs so we won't need to do randomly sampling in the tringing time
        local_dirs = local_dirs[torch.randint(
            local_dirs.shape[0], (N_local_points,), device=self.device)]

        # infer gt sigma & color from teacher model and store them
        mapped_points = mapped_points[mapped_mask]
        mapped_dirs = mapped_dirs[mapped_mask]
        gt_sigma, gt_color = self.teacher_trainer.model(
            mapped_points, mapped_dirs)

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

        # prepare global data and gt
        global_bounds = self.model.aabb_train.view(2, 3)
        global_points, global_dirs = sample_points(
            global_bounds, global_point_step, global_angle_step)
        global_points = global_points.to(
            self.device, torch.float32)
        global_dirs = global_dirs.to(
            self.device, torch.float32)

        _, _, mapped_mask = self.teacher_trainer.model.seal_mapper.map_to_origin(global_points, torch.zeros_like(
            global_points, device=self.device, dtype=torch.float32) + torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32))

        # keep non-edited points
        global_points = global_points[~mapped_mask]
        N_global_points = global_points.shape[0]
        global_dirs = global_dirs[torch.randint(
            global_dirs.shape[0], (N_global_points,), device=self.device)]

        gt_sigma, gt_color = self.teacher_trainer.model(
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


def train(self: trainer_types, train_loader, valid_loader, max_epochs):
    if self.use_tensorboardX and self.local_rank == 0:
        self.writer = tensorboardX.SummaryWriter(
            os.path.join(self.workspace, "run", self.name))

    # mark untrained region (i.e., not covered by any camera from the training dataset)
    if self.model.cuda_ray:
        self.model.mark_untrained_grid(
            train_loader._data.poses, train_loader._data.intrinsics)

    # get a ref to error_map
    self.error_map = train_loader._data.error_map

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

    for epoch in range(self.epoch + 1, max_epochs + 1):
        self.epoch = epoch

        is_pretraining = epoch - first_epoch < self.pretraining_epochs

        if is_pretraining:
            # skip checkpoint saving for pretraining
            self.pretrain_one_epoch()
        else:
            self.freeze_mlp(False)
            self.set_lr(-1)
            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

        if self.epoch % self.eval_interval == 0:
            self.evaluate_one_epoch(valid_loader)
            self.save_checkpoint(full=False, best=True)

    if self.use_tensorboardX and self.local_rank == 0:
        self.writer.close()

# pretrain one epoch. set silent=True to disable logs to speed up, as one epoch of pretraining can be very fast.


def pretrain_one_epoch(self: trainer_types, silent=False):
    # hardcoded lr. not really necessary.
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
    self.pretrain_part('global', silent)
    self.pretrain_part('local', silent)

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

        if not silent and self.scheduler_update_every_step:
            self.lr_scheduler.step()

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

# use both sigma and color to quickly reconstruct modified space


def pretrain_step(self: trainer_types, data):
    # pred_sigma = self.model.density(data['points'])['sigma']
    source = self.pretraining_data[data['source_type']]
    pred_sigma, pred_color = self.model(data['points'], data['dirs'])
    sigma_loss = self.pretraining_criterion(
        pred_sigma, source['sigma'][data['indices'][0]:data['indices'][1]])
    color_loss = self.pretraining_criterion(
        pred_color, source['color'][data['indices'][0]:data['indices'][1]])
    # hardcoded weight. not really necessary as it is just a pretraining.
    loss = color_loss * 100 + sigma_loss
    return loss

# freeze all MLPs or unfreeze them by passing `freeze=False`


def freeze_mlp(self: trainer_types, freeze: bool = True):
    if self._backbone is BackBoneTypes.TensoRF:
        return
    elif self._backbone is BackBoneTypes.NGP:
        if hasattr(self.model, 'sigma_net'):
            freeze_module_list(self.model.sigma_net, freeze)
        if hasattr(self.model, 'color_net'):
            freeze_module_list(self.model.color_net, freeze)
        if hasattr(self.model, 'bg_net') and self.model.bg_net is not None:
            freeze_module_list(self.model.bg_net, freeze)

# manually set learning rate to speedup pretraining. restore the original lr by passing `lr=-1`


def set_lr(self: trainer_types, lr: float):
    if lr < 0:
        if not hasattr(self, '_cached_lr') or self._cached_lr is None:
            return
        lr = self._cached_lr
        self._cached_lr = None
    else:
        self._cached_lr = self.optimizer.param_groups[0]['lr']
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr

# proxy the ground truth RGB from teacher model


def proxy_truth(self: trainer_types, data, all_ray: bool = True, use_cache: bool = False):
    # if the model's bitfield is not hacked, do it before infering
    if not self.teacher_trainer.model.density_bitfield_hacked:
        self.teacher_trainer.model.hack_bitfield()

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
        with torch.no_grad():
            teacher_outputs = self.teacher_trainer.model.render(
                rays_o, rays_d, staged=True, bg_color=None, perturb=False, force_all_rays=all_ray, **vars(self.opt))

    if use_cache:
        if not is_skip_computing:
            self.proxy_cache_image[data['data_index'], data['pixel_index']
                                   [compute_mask]] = torch.nan_to_num(teacher_outputs['image'], nan=0.).detach().cpu()
            self.proxy_cache_depth[data['data_index'], data['pixel_index']
                                   [compute_mask]] = torch.nan_to_num(teacher_outputs['depth'], nan=0.).detach().cpu()
        data['images'] = self.proxy_cache_image[data['data_index'],
                                                data['pixel_index']].to(self.device)
        data['depth'] = self.proxy_cache_depth[data['data_index'],
                                               data['pixel_index']].to(self.device)
    else:
        data['images'] = torch.nan_to_num(teacher_outputs['image'], nan=0.)
        data['depth'] = torch.nan_to_num(teacher_outputs['depth'], nan=0.)
    # reshape if it is a full image
    if is_full:
        data['images'] = data['images'].view(*image_shape[:-1], -1)
        data['depth'] = data['depth'].view(*image_shape[:-1], -1)


def train_step(self: trainer_types, data):
    # if self.teacher_trainer.model.density_bitfield_hacked:
    #     self.teacher_trainer.model.restore_bitfield()
    if self.proxy_train:
        self.proxy_truth(data, use_cache=self.cache_gt)
    return super(self._self, self).train_step(data)


def eval_step(self: trainer_types, data):
    if self.proxy_eval:
        self.proxy_truth(data)
    return super(self._self, self).eval_step(data)


def test_step(self: trainer_types, data, bg_color=None, perturb=False):
    if self.proxy_test:
        self.proxy_truth(data)
    return super(self._self, self).test_step(data, bg_color, perturb)

# this is not a class method


def sample_points(bounds, point_step=0.005, angle_step=45):
    coords_min, coords_max = bounds
    X, Y, Z = torch.meshgrid(torch.arange(coords_min[0], coords_max[0], step=point_step),
                             torch.arange(
        coords_min[1], coords_max[1], step=point_step),
        torch.arange(coords_min[2], coords_max[2], step=point_step))
    sampled_points = torch.stack(
        [X, Y, Z], dim=-1).reshape(-1, 3)

    r_x, r_y, r_z = torch.meshgrid(torch.arange(0, 360, step=angle_step),
                                   torch.arange(0, 360, step=angle_step),
                                   torch.arange(0, 360, step=angle_step))
    eulers = torch.stack([r_x, r_y, r_z], dim=-1).reshape(-1, 3)
    sampled_dirs = torch.from_numpy(Rotation.from_euler('xyz', eulers.numpy(
    ), degrees=True).apply(np.array([1-1e-5, 0, 0])))

    # trimesh.PointCloud(
    #     self.sampled_points.cpu().numpy()).export('tmp/sampled.obj')
    return sampled_points, sampled_dirs

def freeze_module_list(module_list: torch.nn.ModuleList, freeze: bool):
    module_list.training = not freeze
    for i in range(len(module_list)):
        module_list[i].requires_grad_(not freeze)