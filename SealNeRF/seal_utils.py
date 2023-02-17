import os
from typing import Union

import json5
import numpy as np
import torch
import trimesh
from trimesh.creation import uv_sphere
from trimesh.proximity import ProximityQuery
from scipy.optimize import leastsq
from skspatial.objects import Plane


# the virtual root class of all kinds of seal mappers
class SealMapper:
    def __init__(self) -> None:
        self.device = 'cpu'
        self.dtype = torch.float32
        # must be initialized variables in `map_data`:
        # force_fill_bound: for `hack_bitfield`/`hack_grid` in trainer.py
        # map_bound: for `map_mask`
        # pose_center: for pose generation
        # pose_radius: for pose generation
        self.map_data = {}

    # @virtual
    # map the points & dirs back to where they are from
    def map_to_origin(self, points: torch.Tensor, dirs: torch.Tensor = None):
        raise NotImplementedError()

    # convert self.map_data to desired device and dtype
    def map_data_conversion(self, T: torch.Tensor = None, force: bool = False):
        if T is None and not force:
            return
        if T is not None and (self.device != T.device or self.dtype != T.dtype):
            self.device, self.dtype = T.device, T.dtype
        elif not force:
            return
        for k, v in self.map_data.items():
            if isinstance(v, torch.Tensor):
                self.map_data[k] = v.to(
                    device=self.device, dtype=self.dtype)
            elif isinstance(v, np.ndarray):
                self.map_data[k] = torch.from_numpy(v).to(
                    device=self.device, dtype=self.dtype)
            elif isinstance(v, (float, int)):
                self.map_data[k] = torch.tensor(
                    v, device=self.device, dtype=self.dtype)

    # early terminate computation of points outside bbox
    # TODO: if we implement ray contact with mesh/point signed distance to mesh in PyTorch,
    # we could use oriented bbox to minify the search space
    def map_mask(self, points: torch.Tensor) -> torch.BoolTensor:
        return torch.logical_and(points.all(1), torch.logical_and(self.map_data['map_bound'][1] > points, points > self.map_data['map_bound'][0]).all(1))

# seal tool, transform and resize space inside a bbox
# TODO: use oriented bbox
# seal_config format:
# type: bbox
# raw: [N,3] points
# transform: [4,4]
# scale: [3]
class SealBBoxMapper(SealMapper):
    def __init__(self, config_path: str, seal_config: object) -> None:
        super().__init__()

        source_to_target_transform = np.array(seal_config['transform'])
        source_to_target_rotation = np.array(
            seal_config['transform'])[:3, :3]
        source_to_target_scale = np.array(seal_config['scale'])

        # bbox of original points
        self.from_mesh = get_trimesh_box(
            np.array(seal_config['raw']))
        from_center = self.from_mesh.centroid

        # bbox of the target points (points that are to be mapped)
        self.to_mesh: trimesh.Trimesh = self.from_mesh.copy()
        # apply operations to construct `to_mesh` from `from_mesh`
        self.to_mesh.apply_translation(-from_center)
        self.to_mesh.apply_scale(source_to_target_scale)
        self.to_mesh.apply_translation(from_center)
        self.to_mesh.apply_transform(source_to_target_transform)
        to_center = self.to_mesh.centroid

        if self.from_mesh is None or self.to_mesh is None:
            raise RuntimeError('Seal config from_mesh and to_mesh is not set.')
        self.from_mesh.export(os.path.join(config_path, 'from.obj'))
        self.to_mesh.export(os.path.join(config_path, 'to.obj'))

        self.map_data = {
            'force_fill_bound': self.to_mesh.bounds,
            'map_bound': self.to_mesh.bounds,
            'pose_center': (from_center + to_center) / 2,
            'pose_radius': np.linalg.norm(from_center - to_center, 2) * 10,
            # 4 * 4
            'transform': np.linalg.inv(source_to_target_transform),
            # 3 * 3
            'rotation': np.linalg.inv(source_to_target_rotation),
            'scale': 1 / source_to_target_scale,
            'center': from_center
        }
        self.map_data_conversion(force=True)

    @torch.cuda.amp.autocast(enabled=False)
    def map_to_origin(self, points: torch.Tensor, dirs: torch.Tensor = None):
        self.map_data_conversion(points)
        # points & dirs: [N, 3]
        has_dirs = not dirs is None
        map_mask = self.map_mask(points)
        if not map_mask.any():
            return points, dirs, 0

        inner_points = points[map_mask]
        inner_dirs = dirs[map_mask] if has_dirs else None

        N_points, N_dims = inner_points.shape

        origin_inner_points = torch.matmul(self.map_data['transform'], torch.vstack(
            [inner_points.T, torch.zeros([1, N_points], device=inner_points.device)])).T[:, :N_dims]
        origin_inner_dirs = torch.matmul(
            self.map_data['rotation'], inner_dirs.T).T if has_dirs else None
        points_copy = points.clone()
        dirs_copy = dirs.clone() if has_dirs else None

        origin_inner_points = (
            origin_inner_points - self.map_data['center']) * self.map_data['scale'] + self.map_data['center']

        points_copy[map_mask] = origin_inner_points
        if has_dirs:
            dirs_copy[map_mask] = origin_inner_dirs

        # trimesh.PointCloud(
        #     points[inner_points_indices].cpu().numpy()).export('tmp/raw.obj')
        # trimesh.PointCloud(points_copy[inner_points_indices].cpu().numpy()).export(
        #     'tmp/mapped.obj')
        # trimesh.PointCloud(points[~inner_points_indices].cpu().numpy()).export(
        #     'tmp/others.obj')

        return points_copy, dirs_copy, N_points


# brush tool, increase/decrease the surface height along normal direction
# seal_config format:
# type: brush
# raw: [N,3] points
# normal: [3] decide which side of the plane is the positive side
# brushPressure: float maximun height, can be negative
# attenuationDistance: float d(point - center) < attenuationDistance, keeps the highest pressure
# attenuationMode: float d(point - center) > attenuationDistance, pressure attenuates. linear, ease-in, ease-out
class SealBrushMapper(SealMapper):
    def __init__(self, config_path: str, seal_config: object) -> None:
        super().__init__()

        points = seal_config['raw']
        # compute plane
        plane = Plane.best_fit(points)
        # compute normal
        if 'normal' in seal_config and plane.normal @ np.array(seal_config['normal']) < 0:
            plane.normal *= -1

        # generate force filled grids bound
        normal_expand = plane.normal * seal_config['brushPressure']
        bound_mesh = get_trimesh_box(np.vstack([points + normal_expand, points - 0.3 * normal_expand])
                                     )
        # the target space has twice the height of filled bound
        self.to_mesh = get_trimesh_box(np.vstack([points + 2 * normal_expand, points - 0.3 * normal_expand])
                                       )
        self.to_mesh.export(os.path.join(config_path, 'to.obj'))

        # prepare query instance to get brush attenuation
        self.to_mesh_query = ProximityQuery(self.to_mesh)

        self.map_data = {
            'force_fill_bound': bound_mesh.bounds,
            'map_bound': self.to_mesh.bounds,
            'pose_center': bound_mesh.centroid,
            'pose_radius': np.linalg.norm(
                bound_mesh.bounds[1] - bound_mesh.bounds[0], 2) * 10,
            'normal_expand': normal_expand,
            'center': plane.point,
            'attenuation_distance': seal_config['attenuationDistance'],
            'attenuation_mode': seal_config['attenuationMode']
        }
        self.map_data_conversion(force=True)

    def map_to_origin(self, points: Union[torch.Tensor, np.ndarray], dirs: Union[torch.Tensor, np.ndarray] = None):
        self.map_data_conversion(points)
        # TODO: convert dirs for better surface & reflection
        has_dirs = False
        map_mask = self.map_mask(points)
        if not map_mask.any():
            return points, dirs, 0

        inner_points = points[map_mask]
        inner_dirs = dirs[map_mask] if has_dirs else None

        N_points, N_dims = inner_points.shape

        # TODO: implement this in PyTorch
        brush_border_distance = torch.from_numpy(self.to_mesh_query.signed_distance(
            inner_points.cpu().numpy())).to(self.device, dtype=self.dtype)

        points_mapped = inner_points - self.map_data['normal_expand']
        mode = self.map_data['attenuation_mode']
        if mode == 'linear':
            # N_points, 3
            points_compensation = (torch.abs(self.map_data['attenuation_distance'] - brush_border_distance) /
                                   self.map_data['attenuation_distance'])[None].T @ self.map_data['normal_expand'][None]
            points_mapped += points_compensation
        elif mode == 'ease-in':
            # TODO: implement this
            raise NotImplementedError()
        elif mode == 'ease-out':
            # TODO: implement this
            raise NotImplementedError()

        points_copy = points.clone()
        points_copy[map_mask] = points_mapped

        return points_copy, dirs, N_points


# control point (anchor) tool
# seal_config format:
# type: anchor
# raw: [N,3] points, determine the plane
# translation: [3]
# radius: float affected area radius
class SealAnchorMapper(SealMapper):
    def __init__(self, config_path: str, seal_config: object) -> None:
        super().__init__()
        v_translation = np.array(seal_config['translation'])
        len_translation = np.linalg.norm(v_translation, 2)
        v_anchor = np.mean(seal_config['raw'], 0)

        plane = Plane.best_fit(seal_config['raw'])

        v_translated_anchor = v_anchor + v_translation
        v_projected_translated_anchor = plane.project_point(
            v_translated_anchor)
        v_offset = v_projected_translated_anchor - v_anchor
        v_h = v_projected_translated_anchor - v_translated_anchor
        len_h = np.linalg.norm(v_h, 2)

        anchor_sphere_points = uv_sphere(
            seal_config['radius'] * 1.1).vertices + v_anchor
        self.to_mesh = get_trimesh_box(
            np.vstack([anchor_sphere_points, v_anchor + 1.1 * v_translation]))

        self.to_mesh.export(os.path.join(config_path, 'to.obj'))

        self.map_data = {
            'force_fill_bound': self.to_mesh.bounds,
            'map_bound': self.to_mesh.bounds,
            'pose_center': self.to_mesh.centroid,
            'pose_radius': len_translation * 10,
            'v_anchor': v_anchor,
            'v_offset': v_offset,
            'v_h': v_h,
            'len_h': len_h,
            'radius': seal_config['radius']
        }
        self.map_data_conversion(force=True)

    def map_to_origin(self, points: torch.Tensor, dirs: torch.Tensor = None):
        self.map_data_conversion(points)
        # TODO: convert dirs for better surface & reflection
        has_dirs = False
        map_mask = self.map_mask(points)
        if not map_mask.any():
            return points, dirs, 0

        # project points to anchor sphere
        projected_points = self.project_points(
            self.map_data['v_h'], self.map_data['v_anchor'], points)
        v_points_to_plane = projected_points - points
        points_plane_dist = torch.norm(v_points_to_plane, 2, 1)

        # [N_points * 3] scale offsets according to the distance of points and plane
        offset_scale = points_plane_dist.unsqueeze(1) / self.map_data['len_h']
        scaled_offset = offset_scale * self.map_data['v_offset']

        projected_offset_points = projected_points - scaled_offset
        # [N_points] the distance of projected offset points to anchor points
        pop_anchor_dist = torch.norm(
            projected_offset_points - self.map_data['v_anchor'], 2, 1)

        # cone filter
        is_points_in_affected_cone = torch.logical_and(pop_anchor_dist <= self.map_data['radius'], points_plane_dist / (
            self.map_data['radius'] - pop_anchor_dist) < self.map_data['len_h'] / self.map_data['radius'] * 1.1)
        # plane side filter
        is_points_in_valid_side = v_points_to_plane @ self.map_data['v_h'] > 0
        valid_mask = torch.logical_and(
            is_points_in_affected_cone, is_points_in_valid_side)
        # fileter points
        valid_points_plane_dist = points_plane_dist[valid_mask]
        # compute map vector
        v_map = - ((self.map_data['len_h'] - valid_points_plane_dist) / 10)[
            None].T @ self.map_data['v_h'][None] / self.map_data['len_h']

        mapped_points = projected_offset_points[valid_mask] - v_map

        points_copy = points.clone()
        points_copy[valid_mask] = mapped_points

        # trimesh.PointCloud(
        #     points.cpu().numpy()).export('tmp/raw.obj')
        # trimesh.PointCloud(points_copy[valid_mask].cpu().numpy()).export(
        #     'tmp/mapped_to.obj')
        # trimesh.PointCloud(points[valid_mask].cpu().numpy()).export(
        #     'tmp/mapped_from.obj')
        # trimesh.PointCloud(points_copy[~map_mask].cpu().numpy()).export(
        #     'tmp/others.obj')
        # trimesh.PointCloud(projected_offset_points.cpu().numpy()).export(
        #     'tmp/projected.obj')

        return points_copy, dirs, valid_mask.sum()

    def project_points(self, plane_norm: torch.Tensor, plane_point: torch.Tensor, target_points: torch.Tensor):
        v_target_to_plane = target_points - plane_point  # N*3
        v_projection = (v_target_to_plane @ plane_norm).unsqueeze(1) / \
            (plane_norm @ plane_norm) * plane_norm  # N*3
        return target_points - v_projection


def get_seal_mapper(config_path: str) -> SealMapper:
    with open(os.path.join(config_path, 'seal.json'), 'r') as f:
        seal_config = json5.load(f)
    if seal_config['type'] == 'bbox':
        return SealBBoxMapper(config_path, seal_config)
    elif seal_config['type'] == 'brush':
        return SealBrushMapper(config_path, seal_config)
    elif seal_config['type'] == 'anchor':
        return SealAnchorMapper(config_path, seal_config)
    else:
        raise NotImplementedError()


def get_trimesh_box(points) -> trimesh.primitives.Box:
    return trimesh.PointCloud(points).bounding_box_oriented