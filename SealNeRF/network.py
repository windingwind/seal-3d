from typing import Union
from .renderer import SealNeRFTeacherRenderer, SealNeRFStudentRenderder, SealNeRFRenderer
from nerf.network import NeRFNetwork as NGPNetwork
from tensoRF.network import NeRFNetwork as TensoRFNetwork


def get_network(backbone: Union['ngp', 'tensoRF'], character: Union['teacher', 'student']):
    NeRFNetwork = network_constructor(
        backbone_refs[backbone], backbone_funcs[backbone])
    return type(f'NeRFNetwork_{backbone}_{character}', (character_refs[character], NeRFNetwork), {})


def network_constructor(ref, func_names: list):
    funcs = {}
    for k in func_names:
        funcs[k] = getattr(ref, k)
    NeRFNetwork = type('NeRFNetwork', (SealNeRFRenderer,), funcs)
    NeRFNetwork._self = NeRFNetwork
    return NeRFNetwork


backbone_funcs = {
    'ngp': ['__init__', 'forward',
            'density', 'background', 'color', 'get_params'],
    'tensoRF': ['__init__', 'init_one_svd', 'get_sigma_feat', 'get_color_feat', 'forward',
                'density', 'background', 'color', 'density_loss', 'upsample_params', 'upsample_model', 'shrink_model', 'get_params']
}

backbone_refs = {
    'ngp': NGPNetwork,
    'tensoRF': TensoRFNetwork
}

character_refs = {
    'student': SealNeRFStudentRenderder,
    'teacher': SealNeRFTeacherRenderer
}
