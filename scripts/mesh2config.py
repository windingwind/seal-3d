# /bin/python
# usage: `python scripts/mesh2config.py --type bbox --mesh_path /path/to/mesh --save_path /path/to/json`

import json5
import os
import trimesh
import argparse
import numpy as np


def generate_default_editing_tool_config(type: str):
    obj = {}
    obj['type'] = type
    obj['raw'] = []
    if type == 'bbox':
        obj['transform'] = [[1, 0, 0, 0], [
            0, 1, 0, 0,], [0, 0, 1, 0], [0, 0, 0, 1]]
        obj['scale'] = [1, 1, 1]
    elif type == 'brush':
        obj['normal'] = [0, 1, 0]
        obj['brushType'] = 'line'
        obj['brushDepth'] = 0.3
        obj['brushPressure'] = 0.05
        obj['attenuationDistance'] = 0.02
        obj['attenuationMode'] = 'linear'
    elif type == 'anchor':
        obj['translation'] = [0, 0.1, 0]
        obj['radius'] = 0.03
    return obj


def generate_editing_tool_config(mesh_path: str, save_path: str, type: str = None):
    mesh = trimesh.load(mesh_path)
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            obj = json5.load(f)
    else:
        head, tail = os.path.split(save_path)
        os.makedirs(head)
        obj = generate_default_editing_tool_config(type)
    with open(save_path, 'w') as f:
        obj['raw'] = mesh.vertices.tolist()
        json5.dump(obj, f, quote_keys=True)
    print(f'{mesh.vertices.shape[0]} points in total, max={np.max(mesh.vertices, 0)}, mean={np.mean(mesh.vertices, 0)}, min={np.min(mesh.vertices, 0)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None)
    parser.add_argument('--mesh_path', type=str, default='../tmp/mesh.ply')
    parser.add_argument('--save_path', type=str, default='../tmp/editing.json')

    opt = parser.parse_args()

    generate_editing_tool_config(opt.mesh_path, opt.save_path, opt.type)
