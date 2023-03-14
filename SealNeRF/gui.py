# copied from `nerf/gui.py`
# this file is kept here for future demo
# currently it is the same as the original one in `nerf`

import os
import math
import torch
from torch import nn
import numpy as np
import dearpygui.dearpygui as dpg
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.measurements import label
from SealNeRF.types import BackBoneTypes, CharacterTypes
from SealNeRF.provider import SealDataset, SealRandomDataset
# from SealNeRF.gui import NeRFGUI
from SealNeRF.trainer import get_trainer
# from SealNeRF.trainer import SealTrainer, OriginalTrainer
import json5
from functools import partial

OriginalTrainer = get_trainer(BackBoneTypes.NGP, CharacterTypes.Teacher)
SealTrainer = get_trainer(BackBoneTypes.NGP, CharacterTypes.Student)


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        # look at this point
        self.center = np.array([0, 0, 0], dtype=np.float32)
        # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.rot = R.from_quat([1, 0, 0, 0])
        # need to be normalized!
        self.up = np.array([0, 1, 0], dtype=np.float32)

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    @pose.setter
    def pose(self, mat):
        rot = np.eye(4, dtype=np.float32)
        rot[:3,:3] = mat[:3,:3]
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        res = rot @ res
        self.rot = R.from_matrix(mat[:3,:3])
        self.center = res[:3, 3] - mat[:3,3]

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    @intrinsics.setter
    def intrinsics(self, intr):
        focal = intr[0]
        self.fovy = 2 * np.degrees(np.arctan(self.H / (2 * focal)))

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        # why this is side --> ? # already normalized.
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * \
            self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


STATE_PREVIEW = 0
STATE_BRUSH = 1
STATE_TEXTURE = 2
STATE_ANCHOR = 3
STATE_TRAIN = -1

class NeRFGUI:
    def __init__(self, opt, teacher_trainer: OriginalTrainer, trainer: SealTrainer, train_loader=None, debug=True):
        # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.cam_fix = False
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.config = None
        
        self.brush_config = {
            "type": "brush",
            "normal": [1, 0, 0],
            "brushType": [],
            "brushDepth": 0.6,
            "brushPressure": 0.02,
            "attenuationDistance": 0.02,
            "attenuationMode": "dry",
            "simplifyVoxel": 16
        }
        self.brush_thickness = 10
        self.brush_type = "line"
        self.brush_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
        self.brush_mask = []
        self.ii, self.jj = np.meshgrid(np.arange(opt.H), np.arange(opt.W), indexing='ij')
        self.ii, self.jj = self.ii[..., np.newaxis], self.jj[..., np.newaxis]
        # self.training = False
        self.brushing = False

        self.texture_tl = self.texture_br = None
        self.active_texture = None
        self.active_texture_path = None

        self.anchor_uv = []
        self.anchor_points = None
        self.anchor_lookat = None
        self.anchor_dir = [0.0, 0.1, 0.0]
        self.anchor_radius = 0.03

        self.state = STATE_PREVIEW
        self.step = 0  # training step

        self.teacher_trainer = teacher_trainer
        self.trainer = trainer
        self.render_trainer = trainer
        self.train_recorder = torch.cuda.Event(enable_timing=True)
        self.frame_recorder = None
        self.is_pretraining = False
        self.pretrain_only = False
        self.use_time_limit = False
        self.time_limit = 60
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.force_update = False
        self.spp = 1  # sample per pixel
        self.mode = 'image'  # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def train_step(self):

        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # print(type(self.trainer))
        # print(dir(self.trainer))
        outputs = self.trainer.train_gui(
            self.train_loader, step=self.train_steps, is_pretraining=self.is_pretraining)
            # self.train_loader, step=self.train_steps, is_pretraining=self.step > self.trainer.pretraining_epochs)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        if self.is_pretraining and self.step > self.trainer.pretraining_epochs:
            t_pretrain = self.train_recorder.elapsed_time(ender)
            self.trainer.log(f"[INFO] Pretraining time: {t_pretrain/1000:.4f}s")
            # self.is_pretraining = False
            self.force_update = True
            if self.pretrain_only:
                self.state = STATE_PREVIEW
                self.is_pretraining = False
                dpg.configure_item(
                    "_button_train", label="start")
        if self.use_time_limit:
            t_training = self.train_recorder.elapsed_time(ender)
            if t_training / 1000 > self.time_limit:
                self.trainer.log(f"[INFO] Training end with time: {t_training / 1000:.4f}s")
                self.state = STATE_PREVIEW
                dpg.configure_item(
                    "_button_train", label="start")
        self.need_update = True

        # dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        # dpg.set_value(
        #     "_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs):

        def mix_brush(img, mask, color, alpha=1.0):
            mask = np.sum(mask, axis=0)
            return np.where(mask > 0, color[np.newaxis,np.newaxis,:] * alpha + img * (1 - alpha), img)
            # return np.where(mask, color[np.newaxis,np.newaxis,:], color[np.newaxis,np.newaxis,:])
        
        def mix_texture(img, x1, x2, y1, y2, texture):
            if texture.shape[-1] == 3:
                img[y1:y2,x1:x2,:] = texture
                return img
            elif texture.shape[-1] == 4:
                alpha = texture[...,-1:]
                texture = texture[...,:-1]
                crop = img[y1:y2,x1:x2,:]
                img[y1:y2,x1:x2,:] = alpha * texture + (1 - alpha) * crop
                return img
            else:
                raise ValueError()
        
        def mix_anchor(img, uv):
            if len(uv) == 2:
                return cv2.line(outputs['image'], uv[0], uv[1], color=(1,0,0), thickness=2)
            elif len(uv) == 3:
                img = cv2.line(outputs['image'], uv[0], uv[1], color=(1,0,0), thickness=2)
                img = cv2.line(img, uv[0], uv[2], color=(1,0,0), thickness=2)
                img = cv2.line(img, uv[1], uv[2], color=(1,0,0), thickness=2)
                cu = (uv[0][0] + uv[1][0] + uv[2][0]) // 3
                cv = (uv[0][1] + uv[1][1] + uv[2][1]) // 3
                img = cv2.arrowedLine(img, (cu, cv), self.anchor_lookat, (0,0,1), thickness=2)
                return img

        if self.mode == 'image':
            if self.state == STATE_BRUSH:
                return mix_brush(outputs['image'], self.brush_mask + [self.active_mask], self.brush_color)
            elif self.state == STATE_TEXTURE:
                if self.texture_tl is None or self.texture_br is None:
                    return outputs['image']
                if self.active_texture is None:
                    # print(f"Rectangle {self.texture_tl} {self.texture_br}")
                    return cv2.rectangle(outputs['image'], self.texture_tl, self.texture_br, color=(1,0,0), thickness=2)
                else:
                    x1, y1 = self.texture_tl
                    x2, y2 = self.texture_br
                    return mix_texture(outputs['image'], x1, x2, y1, y2, self.active_texture)
            elif self.state == STATE_ANCHOR:
                if len(self.anchor_uv) <= 1:
                    return outputs['image']
                else:
                    return mix_anchor(outputs['image'], self.anchor_uv)
            elif self.state == STATE_TRAIN:
                img = outputs['image']
                # ender = torch.cuda.Event(enable_timing=True)
                # ender.record()
                # torch.cuda.synchronize()
                # t = int(np.round(self.train_recorder.elapsed_time(ender) / 1000))
                if self.is_pretraining:
                    return cv2.putText(img, "Pretrain", (self.W - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,0), thickness=2)
                    # return cv2.putText(img, f"Pretrain: {t}s", (self.W - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,0), thickness=2)
                else:
                    return cv2.putText(img, "Finetune", (self.W - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,0), thickness=2)
                    # return cv2.putText(img, f"Pretrain: {t}s", (self.W - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,0), thickness=2)
            else:
                return outputs['image']
        else:
            # print(outputs['depth'].max())
            depth_img = np.expand_dims(outputs['depth'] / (outputs['depth'].max() + 1e-6), -1).repeat(3, -1)
            if self.state != STATE_BRUSH:
                return depth_img
            else:
                return mix_brush(depth_img, self.brush_mask + [self.active_mask], self.brush_color)
    
    # def get_mask_pos(self, cluster=True):
    #     position = self.teacher_trainer.test_gui(
    #             self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, 1, 1, True)['pos']
    #     if not cluster:
    #         return position[self.brush_mask[:,:,0] > 0]
    #     mask = self.brush_mask[:,:,0]
    #     labeled, ncomponents = label(mask, np.ones((3, 3), dtype=np.uint8))
    #     self.trainer.log(f"[INFO] {ncomponents} brush components found")
    #     return [position[labeled == i+1] for i in range(ncomponents)]
    def get_mask_pos(self, cluster=True):
        position = self.trainer.test_gui(
                self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, 1, 1, True)['pos']
        if cluster:
            return [position[mask[:,:,0] > 0] for mask in self.brush_mask]
        else:
            return position
    
    def update_anchor_lookat(self):
        assert self.state == STATE_ANCHOR and len(self.anchor_points) == 3
        anchor_centroid = np.mean(self.anchor_points, axis=0)
        anchor_lookat = (anchor_centroid + self.anchor_dir).reshape(3, 1)
        xyzw = np.ones((4, 1), dtype=np.float32)
        xyzw[:3,:] = anchor_lookat
        w2c = np.linalg.inv(self.cam.pose)
        fx, fy, cx, cy = self.cam.intrinsics
        K = np.eye(3, dtype=np.float32)
        K[0,0] = fx; K[1,1] = fy; K[0,2] = cx; K[1,2] = cy
        xyzw = (w2c @ xyzw).reshape(4)
        xyz = (xyzw / xyzw[-1])[:3].reshape(3, 1)
        uvw = (K @ xyz).reshape(3)
        u, v = (uvw / uvw[-1])[:2]
        u, v = int(u), int(v)
        u = max(min(u, self.W - 1), 0)
        v = max(min(v, self.H - 1), 0)
        self.anchor_lookat = (u, v)
        self.need_update = True

    
    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?
        if hasattr(self.trainer, 'is_proxy_running') and self.trainer.is_proxy_running.value:
            return

        if self.need_update or self.spp < self.opt.max_spp:

            starter, ender = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.render_trainer.test_gui(
                self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (
                    self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            # dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value(
                "_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            # dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)

    def register_dpg(self):

        # register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer,
                                format=dpg.mvFormat_Float_rgb, tag="_texture")

        # register window

        # the rendered image, as the primary window
        # with dpg.window(tag="_primary_window", width=self.W, height=self.H):
        with dpg.window(tag="_primary_window", autosize=True, no_scrollbar=True, pos=(0,0)):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300, no_close=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(
                        dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(
                        dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            # if not self.opt.test:
            #     with dpg.group(horizontal=True):
            #         dpg.add_text("Train time: ")
            #         dpg.add_text("no data", tag="_log_train_time")

            # with dpg.group(horizontal=True):
            #     dpg.add_text("Infer time: ")
            #     dpg.add_text("no data", tag="_log_infer_time")

            # with dpg.group(horizontal=True):
            #     dpg.add_text("SPP: ")
            #     dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                def callback_file_json(sender, app_data):
                    with open(app_data['file_path_name'], 'r') as f:
                        self.config = json5.load(f)
                    # file_name = app_data['file_name']
                    # current_path = app_data['current_path']
                
                with dpg.file_dialog(directory_selector=False, show=False, callback=callback_file_json, id="_json_selector", width=400,height=300):
                    dpg.add_file_extension(".json")
                # dpg.add_file_dialog(
                #     show=False, callback=callback_file, tag="_json_selector", width=600 ,height=400)

                def callback_file_img(sender, app_data):
                    print(app_data['file_path_name'])
                    texture = cv2.imread(app_data['file_path_name'], -1)
                    if texture.shape[-1] == 4:
                        texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2RGBA)
                    else:
                        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                    texture = (texture.astype(np.float32) / 255)
                    ht = self.texture_br[1] - self.texture_tl[1]
                    wt = self.texture_br[0] - self.texture_tl[0]
                    texture = cv2.resize(texture, (wt, ht), interpolation=cv2.INTER_AREA)
                    self.active_texture = texture
                    self.active_texture_path = app_data['file_path_name']
                    dpg.configure_item("_control_window", collapsed=True)
                    self.need_update = True

                with dpg.file_dialog(directory_selector=False, show=False, callback=callback_file_img, id="_img_selector", width=400,height=300):
                    dpg.add_file_extension(".png")
                    dpg.add_file_extension(".jpg")

                with dpg.collapsing_header(label="Train", default_open=False):
                    with dpg.group(horizontal=True, tag="_train"):
                        dpg.add_text("Train: ")
                        def callback_train(sender, app_data):
                            if self.state == STATE_TRAIN:
                                self.state = STATE_PREVIEW
                                # self.config = None
                                dpg.configure_item(
                                    "_button_train", label="start")
                                return
                            if self.state not in [STATE_BRUSH, STATE_TEXTURE, STATE_ANCHOR] and self.config is None:
                                dpg.set_value("_log_train", "No edit configure!")
                            else:
                                dpg.set_value("_log_train", "")
                                if self.state == STATE_BRUSH:
                                    callback_config_brush(None, None)
                                    dpg.configure_item(
                                        "_button_brush", label="paint")
                                    self.brush_mask = []
                                    self.brush_config['brushType'] = []
                                    self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                                elif self.state == STATE_TEXTURE:
                                    callback_texture_config(None, None)
                                    dpg.configure_item(
                                        "_button_texbox", label="select")
                                    self.active_texture = self.active_texture_path = self.texture_br = self.texture_tl = None
                                elif self.state == STATE_ANCHOR:
                                    callback_anchor_config(None, None)
                                    dpg.configure_item(
                                        "_button_anchor", label="select")
                                    self.anchor_uv = []
                                    self.anchor_points = self.anchor_lookat = None

                                self.trainer.model.init_mapper(self.opt.workspace, self.config)
                                self.teacher_trainer.model.init_mapper(self.opt.workspace, self.config)
                                self.trainer.init_pretraining(epochs=self.opt.pretraining_epochs,
                                    local_point_step=self.opt.pretraining_local_point_step,
                                    local_angle_step=self.opt.pretraining_local_angle_step,
                                    surrounding_point_step=self.opt.pretraining_surrounding_point_step,
                                    surrounding_angle_step=self.opt.pretraining_surrounding_angle_step,
                                    surrounding_bounds_extend=self.opt.pretraining_surrounding_bounds_extend,
                                    global_point_step=self.opt.pretraining_global_point_step,
                                    global_angle_step=self.opt.pretraining_global_angle_step,
                                    batch_size=self.opt.pretraining_batch_size,
                                    lr=self.opt.pretraining_lr)
                                if self.trainer.pretraining_epochs > 0 and self.step <= self.trainer.pretraining_epochs:
                                    self.is_pretraining = True
                                    self.trainer.log(f"[INFO] Pretraining epochs: {self.trainer.pretraining_epochs}")
                                self.train_recorder.record()
                                torch.cuda.synchronize()
                                dpg.configure_item(
                                    "_button_train", label="stop")
                                dpg.configure_item("_control_window", collapsed=True)
                                dpg.focus_item("_primary_window")
                                self.state = STATE_TRAIN
                                self.need_update = True
                        dpg.add_button(
                            label="start", tag="_button_train", callback=callback_train)
                        dpg.add_text(tag="_log_train")
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_pre_only(sender, app_data):
                            self.pretrain_only = not self.pretrain_only
                        dpg.add_checkbox(label="pretrain only", callback=callback_pre_only, default_value=False)

                    with dpg.group(horizontal=True):
                        def callback_time_limit(sender, app_data):
                            self.time_limit = app_data
                        def callback_use_timelimit(sender, app_data):
                            self.use_time_limit = not self.use_time_limit
                        dpg.add_checkbox(label='time limit', callback=callback_use_timelimit, default_value=False)
                        dpg.add_slider_int(tag='_slider_time', callback=callback_time_limit, default_value=self.time_limit, min_value=10, max_value=100)

                    # with dpg.collapsing_header(label="Pretrain options", default_open=False):
                    def callback_set_pretraining_epochs(sender, app_data):
                        self.opt.pretraining_epochs = app_data
                    dpg.add_slider_int(label="epochs", min_value=0, max_value=300, default_value=self.opt.pretraining_epochs, callback=callback_set_pretraining_epochs)
                    
                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            with torch.no_grad():
                                self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value(
                                "_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            # use epoch to indicate different calls.
                            self.trainer.epoch += 1
                        
                        def callback_override(sender, app_data):
                            with torch.no_grad():
                                self.teacher_trainer.model.load_state_dict(self.trainer.model.state_dict())
                                if self.trainer.ema is not None and self.teacher_trainer.ema is not None:
                                    sd = self.trainer.ema.state_dict()
                                    shadow_params = sd['shadow_params']
                                    sd['shadow_params'] = [p.clone().detach() for p in shadow_params]
                                    collected_params = sd['collected_params']
                                    sd['collected_params'] = [p.clone().detach() for p in collected_params]
                                    self.teacher_trainer.ema.load_state_dict(sd)
                                    print("[INFO] EMA loaded")
                                self.teacher_trainer.seal_mapper = None
                                self.teacher_trainer.restore_bitfield()
                                self.trainer.seal_mapper = None
                            self.step = 0
                            self.trainer.has_proxied = False
                            self.need_update = True

                        def callback_ckpt_reset(sender, app_data):
                            with torch.no_grad():
                                self.trainer.model.load_state_dict(self.teacher_trainer.model.state_dict())
                                if self.trainer.ema is not None and self.teacher_trainer.ema is not None:
                                    self.trainer.ema.load_state_dict(self.teacher_trainer.ema.state_dict())
                                    print("[INFO] EMA loaded")
                            self.step = 0
                            self.trainer.has_proxied = False
                            self.need_update = True

                        dpg.add_button(
                            label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)
                        dpg.add_button(
                            label="override", tag="_button_override", callback=callback_override)
                        dpg.bind_item_theme("_button_override", theme_button)
                        dpg.add_button(
                            label="reset", tag="_button_ckpt_reset", callback=callback_ckpt_reset)
                        dpg.bind_item_theme("_button_ckpt_reset", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
              
                    with dpg.group(horizontal=True, tag="_config"):
                        dpg.add_text("Config: ")
                        dpg.add_button(label="open", tag="_button_config", callback=lambda:dpg.show_item("_json_selector"))

                        def callback_save_json(sender, app_data):
                            if self.config is None:
                                dpg.set_value("_log_config", "No edit configure!")
                            else:
                                with open(os.path.join(self.opt.workspace, 'interactive.json'), 'w') as f:
                                    json5.dump(self.config, f, indent=2, quote_keys=True)
                        dpg.add_button(label="save", tag="_button_json", callback=callback_save_json)

                        def callback_reset(sender, app_data):
                            self.state = STATE_PREVIEW
                            self.config = None
                            dpg.configure_item(
                                "_button_brush", label="paint")
                            self.brush_mask = []
                            self.brush_config['brushType'] = []
                            self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                            self.need_update = True
                        
                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.add_text(tag="_log_config")

                        dpg.bind_item_theme("_button_config", theme_button)
                        dpg.bind_item_theme("_button_json", theme_button)
                        dpg.bind_item_theme("_button_reset", theme_button)

                with dpg.collapsing_header(label="Brush", default_open=False):

                    # train / stop
                    with dpg.group(horizontal=True, tag="_brush"):
                        dpg.add_text("Brush: ")

                        # def callback_reset(sender, app_data):
                        #     self.state = STATE_PREVIEW
                        #     self.need_update = True
                        #     dpg.configure_item(
                        #         "_button_brush", label="paint")
                        #     self.trainer.model.load_state_dict(self.teacher_trainer.model.state_dict())
                        #     if self.trainer.ema is not None and self.teacher_trainer.ema is not None:
                        #         self.trainer.ema.load_state_dict(self.teacher_trainer.ema.state_dict())
                        #         print("[INFO] EMA loaded")

                        def callback_brush(sender, app_data):
                            if self.state != STATE_BRUSH:
                                self.state = STATE_BRUSH
                                # self.brush_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                                self.brush_mask = []
                                self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                                self.brush_config['brushType'] = []
                                dpg.configure_item(
                                    "_button_brush", label="cancel")
                                dpg.configure_item("_control_window", collapsed=True)
                                dpg.focus_item("_primary_window")
                            else:
                                self.state = STATE_PREVIEW
                                dpg.configure_item(
                                    "_button_brush", label="paint")
                                # self.brush_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                                self.brush_mask = []
                                self.brush_config['brushType'] = []
                                self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                                # dpg.add_button(label="reset", )
                            # elif self.state == STATE_BRUSH:
                            #     brush_pos = self.get_mask_pos() # (N, 3)
                            #     brush_config = dict(raw=brush_pos, rgb=self.brush_color, **self.brush_config)
                            #     self.trainer.model.init_mapper(self.opt.workspace, brush_config)
                            #     self.teacher_trainer.model.init_mapper(self.opt.workspace, brush_config)
                            #     self.trainer.init_pretraining(epochs=self.opt.pretraining_epochs,
                            #         local_point_step=self.opt.pretraining_local_point_step,
                            #         local_angle_step=self.opt.pretraining_local_angle_step,
                            #         surrounding_point_step=self.opt.pretraining_surrounding_point_step,
                            #         surrounding_angle_step=self.opt.pretraining_surrounding_angle_step,
                            #         surrounding_bounds_extend=self.opt.pretraining_surrounding_bounds_extend,
                            #         global_point_step=self.opt.pretraining_global_point_step,
                            #         global_angle_step=self.opt.pretraining_global_angle_step,
                            #         batch_size=self.opt.pretraining_batch_size,
                            #         lr=self.opt.pretraining_lr)
                            #     torch.cuda.synchronize()
                            #     self.train_recorder.record()
                            #     self.state = STATE_TRAIN
                            #     self.is_pretraining = True
                            #     dpg.configure_item(
                            #         "_button_brush", label="stop")
                            # else:
                            #     self.state = STATE_PREVIEW
                            #     dpg.configure_item(
                            #         "_button_brush", label="edit")
                            self.need_update = True

                        def callback_config_brush(sender, app_data):
                            if not self.state == STATE_BRUSH:
                                return
                            if len(self.brush_mask) == 0:
                                dpg.set_value("_log_train", "No edit configure!")
                                return
                            brush_pos = self.get_mask_pos() # (N, 3)
                            if len(brush_pos) == 0:
                                return
                            if isinstance(brush_pos, list):
                                brush_pos = [x.tolist() for x in brush_pos]
                            # print(brush_pos)
                            norm = np.linalg.norm(self.brush_config['normal'], axis=0)
                            self.brush_config['normal'] = [x / norm for x in self.brush_config['normal']]
                            dpg.configure_item("_slider_nx", default_value=self.brush_config['normal'][0])
                            dpg.configure_item("_slider_ny", default_value=self.brush_config['normal'][1])
                            dpg.configure_item("_slider_nz", default_value=self.brush_config['normal'][2])
                            self.config = dict(raw=brush_pos, rgb=self.brush_color.tolist(), **self.brush_config)
                            # with open(os.path.join(self.opt.workspace, 'interactive.json'), 'w') as f:
                            #     json5.dump(brush_config, f, indent=2, quote_keys=True)
                        
                        def callback_save_mask(sender, app_data):
                            mask = np.zeros((self.H, self.W, 1), dtype=np.bool_)
                            for m in self.brush_mask:
                                mask |= (m > 0)
                            mask = (mask.astype(np.uint8) * 255)
                            painted = (self.render_buffer.clip(0, 1) * 255).astype(np.uint8)
                            cv2.imwrite(f"{self.opt.workspace}/mask.png", mask)
                            cv2.imwrite(f"{self.opt.workspace}/painted.png", painted[:,:,::-1])
                            np.savez(f"{self.opt.workspace}/camera.npz", pose=self.cam.pose, intrinsics=self.cam.intrinsics)

                        dpg.add_button(
                            label="paint", tag="_button_brush", callback=callback_brush)
                        # dpg.add_button(
                        #     label="reset", tag="_button_brush_reset", callback=callback_reset)
                        dpg.add_button(
                            label="config", tag="_button_brush_config", callback=callback_config_brush)
                        dpg.add_button(
                            label="mask", tag="_button_brush_mask", callback=callback_save_mask)
                        dpg.bind_item_theme("_button_brush", theme_button)
                        # dpg.bind_item_theme("_button_brush_reset", theme_button)
                        dpg.bind_item_theme("_button_brush_config", theme_button)
                        dpg.bind_item_theme("_button_brush_mask", theme_button)
                    
                    # mode combo
                    def callback_change_type(sender, app_data):
                        self.brush_type = app_data
                        self.need_update = True

                    dpg.add_combo(('line', 'curve'), label='brush type', tag='_combo_type',
                                default_value=self.brush_type, callback=callback_change_type)
                    
                    def callback_change_brush(sender, app_data):
                        self.brush_color = np.array(app_data[:3], dtype=np.float32) # only need RGB in [0, 1]
                        # self.need_update = True

                    dpg.add_color_edit((255, 0, 0), label="brush Color", width=200, tag="_color_brush", no_alpha=True, callback=
                    callback_change_brush)

                    def callback_set_thickness(sender, app_data):
                        self.brush_thickness = app_data

                    dpg.add_slider_int(label="brush thickness", min_value=1, max_value=50, default_value=self.brush_thickness, callback=callback_set_thickness)

                    # dpg.add_3d_slider(label="brush normal")
                    def callback_set_normal_x(sender, app_data):
                        self.brush_config['normal'][0] = app_data
                    def callback_set_normal_y(sender, app_data):
                        self.brush_config['normal'][1] = app_data
                    def callback_set_normal_z(sender, app_data):
                        self.brush_config['normal'][2] = app_data

                    dpg.add_slider_float(label="brush normal x", tag="_slider_nx", min_value=-1.0, max_value=1.0, default_value=self.brush_config['normal'][0], callback=callback_set_normal_x)
                    dpg.add_slider_float(label="brush normal y", tag="_slider_ny", min_value=-1.0, max_value=1.0, default_value=self.brush_config['normal'][1], callback=callback_set_normal_y)
                    dpg.add_slider_float(label="brush normal z", tag="_slider_nz", min_value=-1.0, max_value=1.0, default_value=self.brush_config['normal'][2], callback=callback_set_normal_z)
                    # dpg.add_slider_float(label='brush pressure', min_value=0.0, max_value=)
                    def callback_set_pressure(sender, app_data):
                        self.brush_config['brushPressure'] = app_data
                    def callback_set_brush_depth(sender, app_data):
                        self.brush_config['brushDepth'] = app_data
                    def callback_set_att_dist(sender, app_data):
                        self.brush_config['attenuationDistance'] = app_data
                    def callback_set_simp_vox(sender, app_data):
                        self.brush_config['simplifyVoxel'] = app_data

                    dpg.add_input_float(label='brush pressure', min_clamped=True, min_value=0.0, max_value=1.0, default_value=self.brush_config['brushPressure'], callback=callback_set_pressure)
                    dpg.add_input_float(label='brush depth', min_clamped=True, min_value=0.0, max_value=2.0, default_value=self.brush_config['brushDepth'], callback=callback_set_brush_depth)
                    dpg.add_input_float(label='attenuation distance', min_clamped=True, min_value=0.0, max_value=0.1, default_value=self.brush_config['attenuationDistance'], callback=callback_set_att_dist)
                    dpg.add_slider_int(label="simpify voxel", min_value=5, max_value=20, default_value=self.brush_config['simplifyVoxel'], callback=callback_set_simp_vox)

                    def callback_set_att_mode(sender, app_data):
                        self.brush_config['attenuationMode'] = app_data
                    dpg.add_combo(("dry", "linear"), label="attenuation mode", default_value=self.brush_config['attenuationMode'], callback=callback_set_att_mode)


                    # save mesh
                    # with dpg.group(horizontal=True):
                    #     dpg.add_text("Marching Cubes: ")

                    #     def callback_mesh(sender, app_data):
                    #         self.trainer.save_mesh(
                    #             resolution=256, threshold=10)
                    #         dpg.set_value(
                    #             "_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                    #         # use epoch to indicate different calls.
                    #         self.trainer.epoch += 1

                    #     dpg.add_button(
                    #         label="mesh", tag="_button_mesh", callback=callback_mesh)
                    #     dpg.bind_item_theme("_button_mesh", theme_button)

                    #     dpg.add_text("", tag="_log_mesh")

                    # with dpg.group(horizontal=True):
                    #     dpg.add_text("", tag="_log_train_log")

                with dpg.collapsing_header(label="Texture", default_open=False):
                    with dpg.group(horizontal=True, tag="_texbox"):
                        dpg.add_text("Area: ")
                        def callback_select_texbox(sender, app_data):
                            if self.state != STATE_TEXTURE:
                                self.state = STATE_TEXTURE
                                dpg.configure_item(
                                    "_button_texbox", label="cancel")
                                dpg.configure_item("_control_window", collapsed=True)
                                dpg.focus_item("_primary_window")
                            else:
                                self.config = None
                                self.state = STATE_PREVIEW
                                dpg.configure_item(
                                    "_button_texbox", label="select")
                                self.need_update = True
                            self.texture_br = self.texture_tl = self.active_texture = self.active_texture_path = None
                        dpg.add_button(
                            label="select", tag="_button_texbox", callback=callback_select_texbox)
                        dpg.bind_item_theme("_button_texbox", theme_button)

                        def callback_texture_config(sender, app_data):
                            if not self.state == STATE_TEXTURE:
                                return
                            if self.texture_br is None or self.texture_tl is None or self.active_texture is None:
                                dpg.set_value('_log_texture', "Need to select area and open texture file first")
                                return
                            pos = self.get_mask_pos(False)
                            x1, y1 = self.texture_tl
                            x2, y2 = self.texture_br
                            brush_config = self.brush_config.copy()
                            brush_config['brushType'] = "line"
                            self.config = dict(
                                raw = pos[y1:y2,x1:x2,:].reshape(-1, 3).tolist(),
                                imageConfig=dict(
                                    path=self.active_texture_path,
                                    o=pos[y1,x1].tolist(),
                                    w=pos[y1,x2].tolist(),
                                    h=pos[y2,x1].tolist()
                                ),
                                **brush_config
                            )
                            
                        dpg.add_button(
                            label="config", tag="_button_texconfig", callback=callback_texture_config)
                        dpg.bind_item_theme("_button_texconfig", theme_button)
                    
                    with dpg.group(horizontal=True, tag="_teximg"):
                        dpg.add_text("Texture: ")

                        def callback_open_texture(sender, app_data):
                            if self.state != STATE_TEXTURE:
                                return
                            if self.texture_br is None or self.texture_tl is None:
                                dpg.set_value('_log_texture', "Need to select area first")
                                return
                            dpg.show_item("_img_selector")

                        dpg.add_button(
                            label="open", tag="_button_texopen", callback=callback_open_texture)
                        dpg.bind_item_theme("_button_texopen", theme_button)
                        dpg.add_text(tag='_log_texture')
                
                with dpg.collapsing_header(label="Anchor", default_open=False):
                    with dpg.group(horizontal=True):

                        def callback_anchor(sender, app_data):
                            if self.state == STATE_ANCHOR:
                                self.state = STATE_PREVIEW
                                self.anchor_points = None
                                self.anchor_uv = []
                                self.anchor_lookat = None
                                dpg.configure_item('_button_anchor', label="select")
                                self.need_update = True
                            else:
                                self.state = STATE_ANCHOR
                                dpg.configure_item("_control_window", collapsed=True)
                                dpg.focus_item("_primary_window")
                                dpg.configure_item('_button_anchor', label="cancel")

                        dpg.add_button(label='select', tag='_button_anchor', callback=callback_anchor)
                        dpg.bind_item_theme("_button_anchor", theme_button)

                        def callback_anchor_config(sender, app_data):
                            if self.state != STATE_ANCHOR or len(self.anchor_uv) < 3:
                                return
                            self.config = dict(
                                raw=self.anchor_points.tolist(),
                                type="anchor",
                                translation=self.anchor_dir,
                                radius=self.anchor_radius,
                                scale=[1,1,1]
                            )
                        dpg.add_button(label='config', tag='_button_config_anchor', callback=callback_anchor_config)
                        dpg.bind_item_theme("_button_config_anchor", theme_button)

                    def callback_set_adir_x(sender, app_data):
                        self.anchor_dir[0] = app_data
                        if len(self.anchor_uv) == 3:
                            self.update_anchor_lookat()
                    def callback_set_adir_y(sender, app_data):
                        self.anchor_dir[1] = app_data
                        if len(self.anchor_uv) == 3:
                            self.update_anchor_lookat()
                    def callback_set_adir_z(sender, app_data):
                        self.anchor_dir[2] = app_data
                        if len(self.anchor_uv) == 3:
                            self.update_anchor_lookat()

                    dpg.add_slider_float(label="anchor direction x", tag="_slider_ax", min_value=-1.0, max_value=1.0, default_value=self.anchor_dir[0], callback=callback_set_adir_x)
                    dpg.add_slider_float(label="anchor direction y", tag="_slider_ay", min_value=-1.0, max_value=1.0, default_value=self.anchor_dir[1], callback=callback_set_adir_y)
                    dpg.add_slider_float(label="anchor direction z", tag="_slider_az", min_value=-1.0, max_value=1.0, default_value=self.anchor_dir[2], callback=callback_set_adir_z)

                    def callback_set_aradius(sender, app_data):
                        self.anchor_radius = app_data
                    dpg.add_input_float(label="anchor radius", tag="_slider_ar", min_clamped=True, min_value=0.0, max_value=0.2, default_value=self.anchor_radius, callback=callback_set_aradius)


            with dpg.collapsing_header(label="Camera", default_open=True):

                def callback_fix(sender, app_data):
                    self.cam_fix = not self.cam_fix
                dpg.add_checkbox(label="fix camera", default_value=self.cam_fix,
                                     callback=callback_fix)
                
                def callback_change_radius(sender, app_data):
                    self.cam.radius = app_data
                    self.need_update = True
                dpg.add_slider_float(label="radius", tag="_slider_radius", min_value=0.1, max_value=3.0, default_value=self.cam.radius, callback=callback_change_radius)
                
                if self.train_loader is not None:
                    cameras = [f"train_{i}" for i in range(len(self.train_loader._data.poses))]
                    def callback_change_camera(sender, app_data):
                        i = int(app_data[6:])
                        # print(self.train_loader._data.poses[i])
                        self.cam.pose = self.train_loader._data.poses[i].cpu().numpy()
                        self.cam.intrinsics = self.train_loader._data.intrinsics
                        # print(self.cam.pose)
                        self.need_update = True
                    dpg.add_combo(cameras, label='camera',
                              default_value="", callback=callback_change_camera)

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=False):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution,
                                     callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth'), label='mode',
                              default_value=self.mode, callback=callback_change_mode)
                
                def callback_change_trainer(sender, app_data):
                    if app_data == 'student':
                        self.render_trainer = self.trainer
                    else:
                        self.render_trainer = self.teacher_trainer
                
                dpg.add_combo(('student', 'teacher'), label='network',
                              default_value='student', callback=callback_change_trainer)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(
                        app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200,
                                   tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120,
                                   format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f",
                                     default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d",
                                   default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0,
                                         format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound,
                                         format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0,
                                         format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound,
                                         format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0,
                                         format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound,
                                         format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        # register camera handler
        def get_mouse_pos_int():
            mx, my = dpg.get_mouse_pos(local=False)
            return (int(mx), int(my))

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]
            if self.state == STATE_BRUSH:
                if not self.brushing:
                    # mx, my = self.brush_pos
                    raise AssertionError()
                    self.brush_mask.append(np.zeros((self.H, self.W, 1), dtype=np.uint8))
                    self.brushing = True
                    # self.brush_mask |= np.sqrt((self.ii - my) ** 2 + (self.jj - mx) ** 2) <= self.brush_thickness
                else:
                    # self.brush_mask = cv2.line(self.brush_mask, self.brush_pos, get_mouse_pos_int(), (1,), self.brush_thickness)
                    self.active_mask = cv2.line(self.active_mask, self.brush_pos, get_mouse_pos_int(), (1,), self.brush_thickness)
                self.brush_pos = get_mouse_pos_int()
                self.need_update = True
            elif self.state == STATE_TEXTURE:
                if not self.brushing:
                    return
                else:
                    x1, y1 = get_mouse_pos_int()
                    x2, y2 = self.texture_tl
                    self.texture_tl = (min(x1, x2), min(y1, y2))
                    self.texture_br = (max(x1, x2), max(y1, y2))
                    # self.texture_br = get_mouse_pos_int()
                self.need_update = True
            elif self.state == STATE_ANCHOR:
                return
            else:
                if self.cam_fix:
                    return
                self.cam.orbit(dx, dy)
                self.need_update = True

                if self.debug:
                    dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if self.cam_fix or not dpg.is_item_focused("_primary_window") or self.state == STATE_BRUSH:
                return

            delta = app_data

            self.cam.scale(delta)
            dpg.configure_item("_slider_radius", default_value=self.cam.radius)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if self.cam_fix or not dpg.is_item_focused("_primary_window") or self.state == STATE_BRUSH:
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_down_brush(sender, app_data):
            if not dpg.is_item_focused("_primary_window") or self.state not in [STATE_BRUSH, STATE_TEXTURE]:
                return
            if self.state == STATE_TEXTURE and self.texture_tl is not None and self.texture_br is not None:
                return
            if not self.brushing:
                # self.brush_mask.append(np.zeros((self.H, self.W, 1), dtype=np.uint8))
                self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                self.brushing = True
                print("[INFO] Brushing start")
                self.brush_pos = get_mouse_pos_int()
                if self.state == STATE_TEXTURE:
                    self.texture_tl = self.brush_pos
            # print(app_data)
            # print("######")
        #     self.last_pos = dpg.get_mouse_pos()

        def callback_release_brush(sender, app_data):
            if self.state not in [STATE_BRUSH, STATE_TEXTURE, STATE_ANCHOR]:
                return
            if self.state == STATE_ANCHOR:
                if not dpg.is_item_focused("_primary_window"):
                    return
                if len(self.anchor_uv) == 3:
                    return
                uv = get_mouse_pos_int()
                print(uv)
                self.anchor_uv.append(uv)
                if len(self.anchor_uv) == 3:
                    pos = self.get_mask_pos(False)
                    self.anchor_points = np.array([
                        pos[i, j] for (j, i) in self.anchor_uv
                    ], dtype=np.float32)
                    self.update_anchor_lookat()
                self.need_update = True
                return
            if not dpg.is_item_focused("_primary_window"):
                self.brushing = False
                self.active_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                print("[INFO] Brushing break")
                return
            self.brushing = False
            if self.state == STATE_BRUSH:
                self.brush_mask.append(self.active_mask)
                self.brush_config['brushType'].append(self.brush_type)
                print("[INFO] Brushing end")
            elif self.state == STATE_TEXTURE:
                print(self.texture_tl, self.texture_br)
        #     print(app_data)
        #     print("******")

        def callback_press_b(sender, app_data):
            # if not dpg.is_item_focused("_primary_window") or self.state != STATE_BRUSH:
            #     return
            if self.brush_type == "line":
                self.brush_type = "curve"
                dpg.set_value('_combo_type', "curve")
            else:
                self.brush_type = "line"
                dpg.set_value('_combo_type', "line")
        
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=callback_down_brush)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_release_brush)
            dpg.add_key_press_handler(key=dpg.mvKey_B, callback=callback_press_b)

        # dpg.create_viewport(title='seal-gui', width=self.W,
        #                     height=self.H + 60, resizable=True, decorated=True)
        dpg.create_viewport(title='seal-gui', width=self.W,
                            height=self.H, resizable=False, decorated=False)

        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,
                                    0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding,
                                    0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            # if self.training:
            if self.state == STATE_TRAIN:
                # self.test_step()
                self.train_step()
                if self.frame_recorder is None:
                    self.frame_recorder = torch.cuda.Event(enable_timing=True)
                    self.frame_recorder.record()
                    self.test_step()
                else:
                    frame_ender = torch.cuda.Event(enable_timing=True)
                    frame_ender.record()
                    torch.cuda.synchronize()
                    t_frame = self.frame_recorder.elapsed_time(frame_ender)
                    # print(t_frame)
                    if t_frame >= 1000 / 2 or self.force_update: # fix 5 frames while training
                        self.frame_recorder.record()
                        self.test_step()
                        if self.force_update:
                            self.force_update = False
                            self.is_pretraining = False
            else:
                self.test_step()
            dpg.render_dearpygui_frame()
