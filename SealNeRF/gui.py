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
from SealNeRF.types import BackBoneTypes, CharacterTypes
from SealNeRF.provider import SealDataset, SealRandomDataset
# from SealNeRF.gui import NeRFGUI
from SealNeRF.trainer import get_trainer
# from SealNeRF.trainer import SealTrainer, OriginalTrainer
import json, json5

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

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

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
STATE_BBOX = 2
STATE_TRAIN = 3

class NeRFGUI:
    def __init__(self, opt, teacher_trainer: OriginalTrainer, trainer: SealTrainer, train_loader=None, debug=True):
        # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.config = None
        self.brush_config = {
            "type": "brush",
            "normal": [0, 1, 0],
            "brushType": "line",
            "brushDepth": 1,
            "brushPressure": 0.01,
            "attenuationDistance": 0.02,
            "attenuationMode": "linear"
        }
        self.brush_thickness = 10
        self.brush_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.brush_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
        self.ii, self.jj = np.meshgrid(np.arange(opt.H), np.arange(opt.W), indexing='ij')
        self.ii, self.jj = self.ii[..., np.newaxis], self.jj[..., np.newaxis]
        # self.training = False
        self.brushing = False
        self.state = STATE_PREVIEW
        self.step = 0  # training step

        self.teacher_trainer = teacher_trainer
        self.trainer = trainer
        self.train_recorder = torch.cuda.Event(enable_timing=True)
        self.is_pretraining = False
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
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
            self.is_pretraining = False
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
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
            return np.where(mask, color[np.newaxis,np.newaxis,:] * alpha + img * (1 - alpha), img)
            # return np.where(mask, color[np.newaxis,np.newaxis,:], color[np.newaxis,np.newaxis,:])
        if self.mode == 'image':
            if self.state != STATE_BRUSH:
                return outputs['image']
            else:
                return mix_brush(outputs['image'], self.brush_mask, self.brush_color)
        else:
            depth_img = np.expand_dims(outputs['depth'], -1).repeat(3, -1).clamp_max(1)
            if self.state != STATE_BRUSH:
                return depth_img
            else:
                return mix_brush(depth_img, self.brush_mask, self.brush_color)
    
    def get_mask_pos(self):
        position = self.teacher_trainer.test_gui(
                self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, 1, 1, True)['pos']
        return position[self.brush_mask[:,:,0] > 0]


    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:

            starter, ender = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.trainer.test_gui(
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

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value(
                "_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)

    def register_dpg(self):

        # register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer,
                                format=dpg.mvFormat_Float_rgb, tag="_texture")

        # register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):

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
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                def callback_file(sender, app_data):
                    with open(app_data['file_path_name'], 'r') as f:
                        self.config = json5.load(f)
                    # file_name = app_data['file_name']
                    # current_path = app_data['current_path']
                
                with dpg.file_dialog(directory_selector=False, show=False, callback=callback_file, id="_file_selector", width=400,height=300):
                    dpg.add_file_extension(".json")
                # dpg.add_file_dialog(
                #     show=False, callback=callback_file, tag="_file_selector", width=600 ,height=400)

                with dpg.collapsing_header(label="Train", default_open=True):
                    with dpg.group(horizontal=True, tag="_train"):
                        dpg.add_text("Train: ")
                        def callback_train(sender, app_data):
                            if self.state == STATE_TRAIN:
                                self.state = STATE_PREVIEW
                                self.config = None
                                dpg.configure_item(
                                    "_button_train", label="train")
                                return
                            if self.config is None:
                                dpg.set_value("_log_train", "No edit configure!")
                            else:
                                dpg.set_value("_log_train", "")
                                if self.state == STATE_BRUSH:
                                    brush_pos = self.get_mask_pos() # (N, 3)
                                    self.config = dict(raw=brush_pos, rgb=self.brush_color, **self.brush_config)
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
                                torch.cuda.synchronize()
                                self.train_recorder.record()
                                self.state = STATE_TRAIN
                                if self.opt.pretraining_epochs > 0:
                                    self.is_pretraining = True
                                dpg.configure_item(
                                    "_button_train", label="stop")
                        dpg.add_button(
                            label="train", tag="_button_train", callback=callback_train)
                        dpg.add_text(tag="_log_train")
                        dpg.bind_item_theme("_button_train", theme_button)

                    with dpg.group(horizontal=True, tag="_config"):
                        dpg.add_text("Config: ")
                        dpg.add_button(label="open", tag="_button_config", callback=lambda:dpg.show_item("_file_selector"))
                        dpg.bind_item_theme("_button_config", theme_button)

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
                                self.brush_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
                                dpg.configure_item(
                                    "_button_brush", label="reset")
                            else:
                                self.state = STATE_PREVIEW
                                dpg.configure_item(
                                    "_button_brush", label="paint")
                                self.brush_mask = np.zeros((self.H, self.W, 1), dtype=np.uint8)
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

                        def callback_save_json(sender, app_data):
                            if not self.state == STATE_BRUSH:
                                return
                            brush_pos = self.get_mask_pos().tolist() # (N, 3)
                            if len(brush_pos) == 0:
                                return
                            brush_config = dict(raw=brush_pos, rgb=self.brush_color.tolist(), **self.brush_config)
                            with open(os.path.join(self.opt.seal_config, 'seal.json'), 'w') as f:
                                json.dump(brush_config, f, indent=2)

                        dpg.add_button(
                            label="paint", tag="_button_brush", callback=callback_brush)
                        # dpg.add_button(
                        #     label="reset", tag="_button_brush_reset", callback=callback_reset)
                        dpg.add_button(
                            label="json", tag="_button_brush_json", callback=callback_save_json)
                        dpg.bind_item_theme("_button_brush", theme_button)
                        # dpg.bind_item_theme("_button_brush_reset", theme_button)
                        dpg.bind_item_theme("_button_brush_json", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value(
                                "_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            # use epoch to indicate different calls.
                            self.trainer.epoch += 1
                        
                        def callback_override(sender, app_data):
                            self.teacher_trainer.model.load_state_dict(self.trainer.model.state_dict())
                            if self.trainer.ema is not None and self.teacher_trainer.ema is not None:
                                self.teacher_trainer.ema.load_state_dict(self.trainer.ema.state_dict())
                                print("[INFO] EMA loaded")

                        dpg.add_button(
                            label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)
                        dpg.add_button(
                            label="override", tag="_button_override", callback=callback_override)
                        dpg.bind_item_theme("_button_override", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    def callback_change_brush(sender, app_data):
                        self.brush_color = np.array(app_data[:3], dtype=np.float32) # only need RGB in [0, 1]
                        # self.need_update = True

                    dpg.add_color_edit((255, 0, 0), label="Brush Color", width=200, tag="_color_brush", no_alpha=True, callback=
                    callback_change_brush)

                    def callback_set_thickness(sender, app_data):
                        self.brush_thickness = app_data

                    dpg.add_slider_int(label="Brush Thickness", min_value=1, max_value=50, format="%d pix", default_value=self.brush_thickness, callback=callback_set_thickness)



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
            if self.state != STATE_BRUSH:

                self.cam.orbit(dx, dy)
                self.need_update = True

                if self.debug:
                    dpg.set_value("_log_pose", str(self.cam.pose))
            
            else:
                if not self.brushing:
                    # mx, my = self.brush_pos
                    self.brushing = True
                    # self.brush_mask |= np.sqrt((self.ii - my) ** 2 + (self.jj - mx) ** 2) <= self.brush_thickness
                else:
                    self.brush_mask = cv2.line(self.brush_mask, self.brush_pos, get_mouse_pos_int(), (1,), self.brush_thickness)
                self.brush_pos = get_mouse_pos_int()
                self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window") or self.state == STATE_BRUSH:
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window") or self.state == STATE_BRUSH:
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_down_brush(sender, app_data):
            if not dpg.is_item_focused("_primary_window") or self.state != STATE_BRUSH:
                return
            # if not self.brushing
            self.brushing = True
            self.brush_pos = get_mouse_pos_int()
            # print(app_data)
            # print("######")
        #     self.last_pos = dpg.get_mouse_pos()

        def callback_release_brush(sender, app_data):
            if not dpg.is_item_focused("_primary_window") or self.state != STATE_BRUSH:
                return
            self.brushing = False
        #     print(app_data)
        #     print("******")
        
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=callback_down_brush)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_release_brush)

        dpg.create_viewport(title='torch-ngp', width=self.W,
                            height=self.H, resizable=False)

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
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
