import torch
from nerf.utils import Trainer as OriginalTrainer


# the trainer of seal nerf main model
class SealTrainer(OriginalTrainer):
    def __init__(self, name, opt, student_model, teacher_trainer, proxy_train=True, proxy_test=False, proxy_eval=False, criterion=None, optimizer=None, ema_decay=None, lr_scheduler=None, metrics=..., local_rank=0, world_size=1, device=None, mute=False, fp16=False, eval_interval=1, max_keep_ckpt=2, workspace='workspace', best_mode='min', use_loss_as_metric=True, report_metric_at_train=False, use_checkpoint="latest", use_tensorboardX=True, scheduler_update_every_step=False):
        super().__init__(name, opt, student_model, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval,
                         max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)
        # use teacher trainer instead of teacher model directly
        # to make sure it's properly initialized, e.g. device
        self.teacher_trainer = teacher_trainer

        # flags indicating the proxy behavior of different stages
        self.proxy_train = proxy_train
        self.proxy_eval = proxy_eval
        self.proxy_test = proxy_test

    # proxy the ground truth RGB from teacher model
    def proxy_truth(self, data, all_ray: bool = True):
        # if the model's bitfield is not hacked, do it before infering
        if not self.teacher_trainer.model.density_bitfield_hacked:
            self.teacher_trainer.model.hack_bitfield()

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        # if we want a full image (B, H, W, C) or just rays (B, N, C)
        is_full = False
        if 'images' in data:
            images = data['images'] # [B, N, 3/4]
            is_full = images.ndim == 4
            image_shape = images.shape
        elif 'images_shape' in data:
            image_shape = data['images_shape']
            is_full = len(image_shape) == 4

        with torch.no_grad():
            teacher_outputs = self.teacher_trainer.model.render(
                rays_o, rays_d, staged=True, bg_color=None, perturb=False, force_all_rays=all_ray, **vars(self.opt))

        data['images'] = teacher_outputs['image']
        # reshape if it is a full image
        if is_full:
            data['images'] = data['images'].view(*image_shape[:-1], -1)

    def train_step(self, data):
        # if self.teacher_trainer.model.density_bitfield_hacked:
        #     self.teacher_trainer.model.restore_bitfield()
        if self.proxy_train:
            self.proxy_truth(data)
        return super().train_step(data)

    def eval_step(self, data):
        if self.proxy_eval:
            self.proxy_truth(data)
        return super().eval_step(data)

    def test_step(self, data, bg_color=None, perturb=False):
        if self.proxy_test:
            self.proxy_truth(data)
        return super().test_step(data, bg_color, perturb)
