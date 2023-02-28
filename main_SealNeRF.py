import torch
import numpy as np
import argparse

from SealNeRF.provider import NeRFDataset, SealDataset
from SealNeRF.gui import NeRFGUI
from SealNeRF.trainer import SealTrainer, OriginalTrainer
from SealNeRF.utils import seed_everything, PSNRMeter, LPIPSMeter

from functools import partial
from loss import huber_loss

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true',
                        help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    # training options
    parser.add_argument('--iters', type=int, default=30000,
                        help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2,
                        help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true',
                        help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    # network backbone options
    parser.add_argument('--fp16', action='store_true',
                        help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true',
                        help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    # dataset options
    parser.add_argument('--color_space', type=str, default='srgb',
                        help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33,
                        help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*',
                        default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2,
                        help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")
    
    # seal options
    # pretraining strategy
    parser.add_argument('--pretraining_epochs', type=int, default=150,
                        help="num epochs for local pretraining")
    parser.add_argument('--pretraining_point_step', type=float, default=0.001,
                        help="pretraining point sampling step")
    parser.add_argument('--pretraining_angle_step', type=float, default=45,
                        help="pretraining angle sampling step in degree")
    parser.add_argument('--pretraining_batch_size', type=int, default=6144000,
                        help="pretraining angle sampling step in degree")
    # wether to use generated camera poses rotating the seal_config's pose_center within pose_radius
    parser.add_argument('--custom_pose', action='store_true',
                        help="use generated poses")

    # GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5,
                        help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50,
                        help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64,
                        help="GUI rendering max sample per pixel")

    # experimental
    parser.add_argument('--error_map', action='store_true',
                        help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='',
                        help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    # teacher model
    parser.add_argument('--teacher_workspace', type=str,
                        default='workspace', help="teacher trainer workspace")
    parser.add_argument('--teacher_ckpt', type=str, default='latest')
    parser.add_argument('--seal_config', type=str, default='')

    parser.add_argument('--eval_interval', type=int, default=50, help="eval_interval")
    parser.add_argument('--eval_count', type=int, default=10, help="eval_count")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (
            opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    # if opt.ff:
    #     opt.fp16 = True
    #     assert opt.bg_radius <= 0, "background model is not implemented for --ff"
    #     from nerf.network_ff import NeRFNetwork
    # elif opt.tcnn:
    #     opt.fp16 = True
    #     assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
    #     from nerf.network_tcnn import NeRFNetwork
    # else:
    #     # from nerf.network import NeRFNetwork
    #     from SealNeRF.network import SelaNeRFStudentNetwork

    from SealNeRF.network import SealNeRFTeacherNetwork, SelaNeRFStudentNetwork

    print(opt)

    seed_everything(opt.seed)

    model = SelaNeRFStudentNetwork(
        opt.seal_config,
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    print(model)

    teacher_model = SealNeRFTeacherNetwork(
        opt.seal_config,
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    teacher_model.train(False)

    print(teacher_model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    # criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = OriginalTrainer('ngp', opt, model, device=device, workspace=opt.workspace,
                                  criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(
                opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                # blender has gt, so evaluate it.
                trainer.evaluate(test_loader)

            trainer.test(test_loader, write_video=True)  # test and save video

            trainer.save_mesh(resolution=256, threshold=10)

    else:

        def optimizer(model): return torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        # decay to 0.1 * init_lr at last iter step
        def scheduler(optimizer): return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        teacher_trainer = OriginalTrainer('ngp', opt, teacher_model, device=device, workspace=opt.teacher_workspace, optimizer=optimizer, criterion=criterion,
                                          ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.teacher_ckpt, eval_interval=50)
        trainer = SealTrainer('ngp', opt, model, teacher_trainer, proxy_eval=True,
                              pretraining_epochs=opt.pretraining_epochs,
                              pretraining_point_step=opt.pretraining_point_step,
                              pretraining_angle_step=opt.pretraining_angle_step,
                              pretraining_batch_size=opt.pretraining_batch_size,
                              device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95,
                              fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, eval_count=opt.eval_count, max_keep_ckpt=65535)

        if opt.custom_pose:
            train_dataset = SealDataset(
                opt, seal_mapper=model.seal_mapper, device=device, type='train')
            train_loader = train_dataset.dataloader()
            trainer.log(f'[INFO] Dataset: center={train_dataset.look_at}&radius={train_dataset.radius}')
        else:
            train_loader = NeRFDataset(
                opt, device=device, type='train').dataloader()

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            if opt.custom_pose:
                valid_loader = SealDataset(
                    opt, seal_mapper=model.seal_mapper, device=device, type='val').dataloader()
            else:
                valid_loader = NeRFDataset(
                    opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.log(f'[INFO] Proxy train/eval/test: {trainer.proxy_train}/{trainer.proxy_eval}/{trainer.proxy_test}')

            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(
                opt, device=device, type='test').dataloader()

            # teacher_trainer.test(test_loader, write_video=False)

            # if test_loader.has_gt:
            #     trainer.evaluate(test_loader) # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video

            trainer.save_mesh(resolution=256, threshold=10)
