import os
from typing import Union, Tuple
import json5
import numpy as np
import torch
from pytorch3d.structures import Meshes
# must be imported after `import torch`
from pytorch3d import _C
import trimesh
from trimesh.creation import uv_sphere
from skspatial.objects import Plane
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from .color_utils import rgb2hsv_torch, hsv2rgb_torch, rgb2hsl_torch, hsl2rgb_torch


class SealMapper:
    """
    the virtual root class of all kinds of seal mappers
    """

    def __init__(self) -> None:
        self.device = 'cpu'
        self.dtype = torch.float32
        # variables in `map_data`:
        # force_fill_bound: for `hack_bitfield`/`hack_grid` in trainer.py
        # map_bound: for map_mask
        # pose_center: for pose generation
        # pose_radius: for pose generation
        # hsv?: for color modification
        self.map_data = {}
        self.map_meshes: Meshes = None
        self.map_triangles: torch.Tensor = None

    def map_to_origin(self, points: torch.Tensor, dirs: torch.Tensor = None):
        """
        @virtual

        map the points & dirs back to where they are from
        """
        raise NotImplementedError()

    def map_color(self, colors: torch.Tensor) -> torch.Tensor:
        """
        map color
        """
        if 'hsv' in self.map_data:
            colors = modify_hsv(
                colors, self.map_data['hsv'])
        if 'rgb' in self.map_data:
            colors = modify_rgb(
                colors, self.map_data['rgb'])
        return colors

    def map_data_conversion(self, T: torch.Tensor = None, force: bool = False):
        """
        convert self.map_data to desired device and dtype
        """
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
            elif isinstance(v, list):
                self.map_data[k] = torch.from_numpy(np.asarray(v)).to(
                    device=self.device, dtype=self.dtype)
            elif isinstance(v, (float, int)):
                self.map_data[k] = torch.tensor(
                    v, device=self.device, dtype=self.dtype)
        self.map_meshes = self.map_meshes.to(self.device)
        self.map_triangles = self.map_triangles.to(
            device=self.device, dtype=self.dtype)

    def map_mask(self, points: torch.Tensor) -> torch.BoolTensor:
        """
        early terminate computation of points outside bbox
        """
        bound_mask = torch.logical_and(points.all(1), torch.logical_and(
            self.map_data['map_bound'][1] > points, points > self.map_data['map_bound'][0]).all(1))
        if not bound_mask.any():
            return bound_mask
        shape_mask = points_in_mesh(points[bound_mask], self.map_triangles)
        bound_mask[bound_mask.clone()] = shape_mask
        return bound_mask


class SealBBoxMapper(SealMapper):
    """
    seal tool, transform and resize space inside a bbox
    seal_config format:
    type: bbox
    raw: [N,3] points
    transform: [4,4]
    scale: [3]
    """

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

        self.map_meshes = trimesh_to_pytorch3d(self.to_mesh)
        self.map_triangles = self.map_meshes.verts_packed()[
            self.map_meshes.faces_packed()]

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
        if 'hsv' in seal_config:
            self.map_data['hsv'] = seal_config['hsv']
        if 'rgb' in seal_config:
            self.map_data['rgb'] = seal_config['rgb']
        self.map_data_conversion(force=True)

    @torch.cuda.amp.autocast(enabled=False)
    def map_to_origin(self, points: torch.Tensor, dirs: torch.Tensor = None):
        self.map_data_conversion(points)
        # points & dirs: [N, 3]
        has_dirs = not dirs is None
        map_mask = self.map_mask(points)
        if not map_mask.any():
            return points, dirs, map_mask

        inner_points = points[map_mask]
        inner_dirs = dirs[map_mask] if has_dirs else None

        N_points, N_dims = inner_points.shape

        transformed_inner_points = torch.matmul(self.map_data['transform'], torch.vstack(
            [inner_points.T, torch.ones([1, N_points], device=inner_points.device)])).T[:, :N_dims]

        origin_inner_points = (
            transformed_inner_points - self.map_data['center']) * self.map_data['scale'] + self.map_data['center']

        origin_inner_dirs = torch.matmul(
            self.map_data['rotation'], inner_dirs.T).T if has_dirs else None
        points_copy = points.clone()
        dirs_copy = dirs.clone() if has_dirs else None

        points_copy[map_mask] = origin_inner_points
        if has_dirs:
            dirs_copy[map_mask] = origin_inner_dirs

        # trimesh.PointCloud(
        #     points[map_mask].cpu().numpy()).export('tmp/raw.obj')
        # trimesh.PointCloud(points_copy[map_mask].cpu().numpy()).export(
        #     'tmp/mapped.obj')
        # trimesh.PointCloud(points[~map_mask].cpu().numpy()).export(
        #     'tmp/others.obj')

        return points_copy, dirs_copy, map_mask


class SealBrushMapper(SealMapper):
    """
    brush tool, increase/decrease the surface height along normal direction
    seal_config format:
    type: brush
    raw: [N,3] points
    normal: [3] decide which side of the plane is the positive side
    brushType: 'line' | 'curve'
    brushDepth: float maximun affected depth along the opposite direction of normal
    brushPressure: float maximun height, can be negative
    attenuationDistance: float d(point - center) < attenuationDistance, keeps the highest pressure
    attenuationMode: float d(point - center) > attenuationDistance, pressure attenuates. linear, ease-in, ease-out
    """

    def __init__(self, config_path: str, seal_config: object) -> None:
        super().__init__()

        points = np.asarray(seal_config['raw'])
        # compute plane
        plane = Plane.best_fit(points)
        # compute normal
        if 'normal' in seal_config and plane.normal @ np.array(seal_config['normal']) < 0:
            plane.normal *= -1

        # generate force filled grids bound
        normal_expand = plane.normal * seal_config['brushPressure']
        projected_points = project_points(
            torch.from_numpy(plane.normal), torch.from_numpy(plane.point), torch.from_numpy(points))
        if seal_config['brushType'] == 'line':
            self.to_mesh = get_trimesh_box(np.vstack([points + 2 * normal_expand, points - seal_config['brushDepth'] * normal_expand])
                                           )
        else:
            # project to plane so the mesh is smooth
            self.to_mesh = get_trimesh_fit(
                projected_points.numpy(),
                normal_expand, [-seal_config['brushDepth'], 2])
        self.to_mesh.export(os.path.join(config_path, 'to.obj'))

        self.map_meshes = trimesh_to_pytorch3d(self.to_mesh)
        self.map_triangles = self.map_meshes.verts_packed()[
            self.map_meshes.faces_packed()]

        border_points_mask = mesh_surface_points_mask(
            self.map_triangles.to(self.dtype), projected_points.to(self.dtype))
        # trimesh.PointCloud(points[border_points_mask]).export('./tmp/border.ply')

        self.map_data = {
            'force_fill_bound': self.to_mesh.bounds,
            'map_bound': self.to_mesh.bounds,
            'pose_center': self.to_mesh.centroid,
            'pose_radius': np.linalg.norm(
                self.to_mesh.bounds[1] - self.to_mesh.bounds[0], 2) * 10,
            'normal_expand': normal_expand,
            'center': plane.point,
            'border_points': projected_points[border_points_mask],
            'attenuation_distance': seal_config['attenuationDistance'],
            'attenuation_mode': seal_config['attenuationMode']
        }
        if 'hsv' in seal_config:
            self.map_data['hsv'] = seal_config['hsv']
        if 'rgb' in seal_config:
            self.map_data['rgb'] = seal_config['rgb']
        self.map_data_conversion(force=True)

    def map_to_origin(self, points: Union[torch.Tensor, np.ndarray], dirs: Union[torch.Tensor, np.ndarray] = None):
        self.map_data_conversion(points)
        # TODO: convert dirs for better surface & reflection
        has_dirs = False
        map_mask = self.map_mask(points)
        if not map_mask.any():
            return points, dirs, map_mask

        inner_points = points[map_mask]
        inner_dirs = dirs[map_mask] if has_dirs else None

        N_points, N_dims = inner_points.shape

        projected_points = project_points(
            self.map_data['normal_expand'], self.map_data['center'], inner_points)
        brush_border_distance = torch.cdist(
            projected_points, self.map_data['border_points']).min(1)[0]
        # brush_border_distance = points_mesh_distance(
        #     projected_points, self.map_meshes, self.map_triangles)

        mode = self.map_data['attenuation_mode']
        if mode == 'linear':
            points_mapped = inner_points - self.map_data['normal_expand']
            # N_points, 3
            distance_filter = self.map_data['attenuation_distance'] > brush_border_distance
            points_compensation = (torch.abs(self.map_data['attenuation_distance'] - brush_border_distance[distance_filter]) /
                                   self.map_data['attenuation_distance'])[None].T @ self.map_data['normal_expand'][None]
            points_mapped[distance_filter] += points_compensation
        elif mode == 'dry':
            # for dry brush, no space mapping is applied.
            points_mapped = inner_points
        elif mode == 'ease-in':
            # TODO: implement this
            raise NotImplementedError()
        elif mode == 'ease-out':
            # TODO: implement this
            raise NotImplementedError()

        points_copy = points.clone()
        points_copy[map_mask] = points_mapped

        # trimesh.PointCloud(
        #     points.cpu().numpy()).export('tmp/raw.obj')
        # trimesh.PointCloud(points_copy[map_mask].cpu().numpy()).export(
        #     'tmp/mapped_to.obj')
        # trimesh.PointCloud(points[map_mask].cpu().numpy()).export(
        #     'tmp/mapped_from.obj')

        return points_copy, dirs, map_mask


class SealAnchorMapper(SealMapper):
    """
    control point (anchor) tool
    seal_config format:
    type: anchor
    raw: [N,3] points, determine the plane
    translation: [3]
    radius: float affected area radius
    """

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

        self.map_meshes = trimesh_to_pytorch3d(self.to_mesh)
        self.map_triangles = self.map_meshes.verts_packed()[
            self.map_meshes.faces_packed()]

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
        if 'hsv' in seal_config:
            self.map_data['hsv'] = seal_config['hsv']
        if 'rgb' in seal_config:
            self.map_data['rgb'] = seal_config['rgb']
        self.map_data_conversion(force=True)

    def map_to_origin(self, points: torch.Tensor, dirs: torch.Tensor = None):
        self.map_data_conversion(points)
        # TODO: convert dirs for better surface & reflection
        has_dirs = False
        map_mask = self.map_mask(points)
        if not map_mask.any():
            return points, dirs, map_mask

        # project points to anchor sphere
        projected_points = project_points(
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

        return points_copy, dirs, valid_mask


def get_seal_mapper(config_path: str, config_dict: dict = None, config_file: str = 'seal.json') -> SealMapper:
    if config_dict is None:
        with open(os.path.join(config_path, config_file), 'r') as f:
            config_dict = json5.load(f)
    if config_dict['type'] == 'bbox':
        return SealBBoxMapper(config_path, config_dict)
    elif config_dict['type'] == 'brush':
        return SealBrushMapper(config_path, config_dict)
    elif config_dict['type'] == 'anchor':
        return SealAnchorMapper(config_path, config_dict)
    else:
        raise NotImplementedError()


def get_trimesh_box(points) -> trimesh.primitives.Box:
    return trimesh.PointCloud(points).bounding_box_oriented


def get_trimesh_fit(points, normal, growth=[-0.3, 1]) -> trimesh.Trimesh:
    N = points.shape[0]
    K = 10

    neigh = NearestNeighbors(n_neighbors=K, radius=0.4)
    neigh.fit(points)
    indices = neigh.kneighbors(points, K, return_distance=False)

    faces = []

    for i in range(N):
        for j in range(1, K):
            for k in range(j+1, K):
                x, y, z = i, indices[i][j], indices[i][k]
                _x, _y, _z = x + N, y + N, z + N
                faces.append([x, y, z])
                faces.append([_x, _y, _z])
                faces.append([x, y, _x])
                faces.append([_x, y, _y])

    generated_mesh = trimesh.Trimesh(np.concatenate(
        [points + normal * growth[0], points + normal * growth[1]]), faces)

    o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
        generated_mesh.vertices), o3d.utility.Vector3iVector(generated_mesh.faces))

    voxel_size = max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()) / 16
    simplified_mesh = o3d_mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    return trimesh.Trimesh(np.asarray(simplified_mesh.vertices), np.asarray(simplified_mesh.triangles))


def trimesh_to_pytorch3d(mesh: trimesh.Trimesh) -> Meshes:
    return Meshes(torch.from_numpy(mesh.vertices)[None], torch.from_numpy(mesh.faces)[None])


def moller_trumbore(ray_o: torch.Tensor, ray_d: torch.Tensor, tris: torch.Tensor, eps=1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    https://github.com/facebookresearch/pytorch3d/issues/343
    The Moller Trumbore algorithm for fast ray triangle intersection
    Naive batch implementation (m rays and n triangles at the same time)
    O(n_rays * n_faces) memory usage, parallelized execution
    Parameters
    ----------
    ray_o : torch.Tensor, (n_rays, 3)
    ray_d : torch.Tensor, (n_rays, 3)
    tris  : torch.Tensor, (n_faces, 3, 3)
    """
    E1 = tris[:, 1] - tris[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
    E2 = tris[:, 2] - tris[:, 0]  # vector of edge 2 on triangle (n_faces, 3)

    # batch cross product
    # normal to E1 and E2, automatically batched to (n_faces, 3)
    N = torch.cross(E1, E2)

    invdet = 1. / -(torch.einsum('md,nd->mn', ray_d, N) +
                    eps)  # inverse determinant (n_faces, 3)

    # (n_rays, 3) - (n_faces, 3) -> (n_rays, n_faces, 3) automatic broadcast
    A0 = ray_o[:, None] - tris[None, :, 0]
    # (n_rays, n_faces, 3) x (n_rays, 3) -> (n_rays, n_faces, 3) no automatic broadcast
    DA0 = torch.cross(A0, ray_d[:, None].expand(*A0.shape))

    u = torch.einsum('mnd,nd->mn', DA0, E2) * invdet
    v = -torch.einsum('mnd,nd->mn', DA0, E1) * invdet
    t = torch.einsum('mnd,nd->mn', A0, N) * \
        invdet  # t >= 0.0 means this is a ray

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection.any(1)


def points_in_mesh(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    points: <num>[P, 3]
    triangles: <num>[F, 3, 3]
    return: <bool>[P,]
    """
    rays_o = torch.concat([points, points])

    # magic number from `trimesh.Trimesh.contains_points`.
    # the rays_d can be any. use the same ray direction for debug.
    rays_d = torch.tensor([[0.4395064455,
                            0.617598629942,
                            0.652231566745]], device=points.device)
    rays_d = rays_d.repeat(points.shape[0], 1)
    rays_d = torch.concat([rays_d, -rays_d])

    mask = moller_trumbore(rays_o, rays_d, triangles)
    return torch.bitwise_and(mask[:points.shape[0]], mask[-points.shape[0]:])


def points_mesh_distance(points: torch.Tensor, meshes: Meshes, tris: torch.Tensor = None) -> torch.Tensor:
    """
    https://github.com/facebookresearch/pytorch3d/issues/193
    points: <num>[P, 3]
    triangles: pytorch3d.structures.Meshes
    return: <float>[P,]
    """
    # computing tris is time consuming. we can prepare it in advance
    if tris is None:
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    _DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3
    dists, idxs = _C.point_face_dist_forward(
        points, torch.tensor(
            [0], device=points.device), tris, tris_first_idx, points.shape[0], _DEFAULT_MIN_TRIANGLE_AREA
    )

    dists = torch.sqrt(dists)
    return dists


def mesh_surface_points_mask(triangles: torch.Tensor, points: torch.Tensor):
    # offset_value = np.linalg.norm(points.max(0) - points.min(0), 2) / 100
    offset_value = 1e-4
    offsets = torch.from_numpy(np.array([
        [0, 0, offset_value],
        [0, 0, -offset_value],
        [0, offset_value, 0],
        [0, -offset_value, 0],
        [offset_value, 0, 0],
        [-offset_value, 0, 0]
    ])).to(points.device, points.dtype)
    masks = torch.sum(torch.stack([~points_in_mesh(
        points + offsets[i], triangles) for i in range(offsets.shape[0])]), 0) > 0
    return masks


def project_points(plane_norm: torch.Tensor, plane_point: torch.Tensor, target_points: torch.Tensor):
    """
    project 3d points to a plane defined by normal and plane point
    returns: projected points
    """
    v_target_to_plane = target_points - plane_point  # N*3
    v_projection = (v_target_to_plane @ plane_norm).unsqueeze(1) / \
        (plane_norm @ plane_norm) * plane_norm  # N*3
    return target_points - v_projection


def modify_hsv(rgb: torch.Tensor, modification: torch.Tensor):
    """
    rgb -> hsv + mod -> rgb
    """
    N = rgb.shape[0]
    if N == 0:
        return rgb
    hsv = rgb2hsv_torch(rgb.view(N, 3, 1))
    hsv[:, 0, :] += modification[0]
    hsv[:, 1, :] += modification[1]
    hsv[:, 2, :] += modification[2]
    return hsv2rgb_torch(hsv).view(N, 3)


def modify_rgb(rgb: torch.Tensor, modification: torch.Tensor):
    """
    the original color is not correct makes the converted hsl value meaningless
    """
    N = rgb.shape[0]
    if N == 0:
        return rgb
    # return modification.repeat(N).view(N, 3).to(rgb.device, rgb.dtype)
    hsl = rgb2hsl_torch(rgb.view(N, 3, 1))
    hsl_modification = rgb2hsl_torch(modification.view(
        1, 3, 1)).view(3).to(rgb.device, rgb.dtype)
    hsl[:, 0, :] = hsl_modification[0]
    hsl[:, 1, :] = hsl_modification[1]
    ret = hsl2rgb_torch(hsl).view(N, 3)
    return ret
