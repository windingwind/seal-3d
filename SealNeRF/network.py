from .types import BackBoneTypes, CharacterTypes
from .renderer import SealNeRFTeacherRenderer, SealNeRFStudentRenderder, SealNeRFRenderer
from nerf.network import NeRFNetwork as NGPNetwork
from tensoRF.network import NeRFNetwork as TensoRFNetwork


def get_network(backbone: BackBoneTypes, character: CharacterTypes):
    """
    Get network class of `backbone` and `character` defined in `./types.py`
    """
    NeRFNetwork = network_constructor(
        backbone_refs[backbone], backbone_funcs[backbone])
    network_type = type(f'NeRFNetwork_{backbone.name}_{character.name}',
                        (character_refs[character], NeRFNetwork), {})
    return network_type


def network_constructor(ref, func_names: list):
    """
    Construct network class with class methods dynamically copied from `ref`.
    Hack the base class to our renderer.
    """
    funcs = {}
    for k in func_names:
        funcs[k] = getattr(ref, k)
    NeRFNetwork = type('NeRFNetwork', (SealNeRFRenderer,), funcs)
    NeRFNetwork._self = NeRFNetwork
    return NeRFNetwork


backbone_funcs = {
    BackBoneTypes.NGP: ['__init__', 'forward',
                        'density', 'background', 'color', 'get_params'],
    BackBoneTypes.TensoRF: ['__init__', 'init_one_svd', 'get_sigma_feat', 'get_color_feat', 'forward',
                            'density', 'background', 'color', 'density_loss', 'upsample_params', 'upsample_model', 'shrink_model', 'get_params']
}

backbone_refs = {
    BackBoneTypes.NGP: NGPNetwork,
    BackBoneTypes.TensoRF: TensoRFNetwork
}

character_refs = {
    CharacterTypes.Student: SealNeRFStudentRenderder,
    CharacterTypes.Teacher: SealNeRFTeacherRenderer
}
