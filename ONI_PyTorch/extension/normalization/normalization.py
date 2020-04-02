import argparse
import torch.nn as nn


from .NormedConv import IdentityModule, WN_Conv2d, OWN_Conv2d, ONI_Conv2d, ONI_ConvTranspose2d, ONI_Linear
from ..utils import str2dict


def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)


def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _BatchNorm(num_features, dim=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    return (nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d)(num_features, eps=eps, momentum=momentum, affine=affine,
                                                            track_running_stats=track_running_stats)


def _InstanceNorm(num_features, dim=4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, *args,
                  **kwargs):
    return (nn.InstanceNorm2d if dim == 4 else nn.InstanceNorm1d)(num_features, eps=eps, momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats)

def _Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *args, **kwargs):
    """return first input"""
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()

def _Identity_fn(x, *args, **kwargs):
    """return first input"""
    return x


class _config:
    norm = 'BN'
    norm_cfg = {}
    norm_methods = {'No': _IdentityModule, 'BN': _BatchNorm, 'None': None
                    }

    normConv = 'ONI'
    normConv_cfg = {}
    normConv_methods = {'No': _Conv2d, 'WN': WN_Conv2d, 'OWN': OWN_Conv2d,
                        'ONI': ONI_Conv2d}

def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Normalization Options')
    group.add_argument('--norm', default='No', help='Use which normalization layers? {' + ', '.join(
        _config.norm_methods.keys()) + '}' + ' (defalut: {})'.format(_config.norm))
    group.add_argument('--normConv', default='No', help='Use which weight normalization layers? {' + ', '.join(
        _config.normConv_methods.keys()) + '}' + ' (defalut: {})'.format(_config.normConv))
    group.add_argument('--normConv-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    return group

def getNormConfigFlag():
    flag = ''
    flag += _config.norm

    flag += '_' + _config.normConv
    if _config.normConv == 'ONI' or _config.normConv == 'SN' or _config.normConv == 'NSN':
        if _config.normConv_cfg.get('T') != None:
            flag += '_T' + str(_config.normConv_cfg.get('T'))

    if _config.normConv == 'ONI' or str.find(_config.normConv, 'OWN') > -1:
        if _config.normConv_cfg.get('norm_groups') != None:
            flag += '_G' + str(_config.normConv_cfg.get('norm_groups'))
    if _config.normConv == 'ONI' or str.find(_config.normConv, 'CWN') > -1 \
           or _config.normConv == 'Pearson' or _config.normConv == 'WN' or str.find(_config.normConv, 'OWN') > -1:
        if _config.normConv_cfg.get('NScale') != None:
            flag += '_NS' + str(_config.normConv_cfg.get('NScale'))
        if _config.normConv_cfg.get('adjustScale') == True:
            flag += '_AS'
    return flag

def setting(cfg: argparse.Namespace):
    print(_config.__dict__)
    for key, value in vars(cfg).items():
        #print(key)
        #print(value)
        if key in _config.__dict__:
            setattr(_config, key, value)
    #print(_config.__dict__)
    flagName =  getNormConfigFlag()
    print(flagName)
    return flagName


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == 'None':
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)

def NormConv(*args, **kwargs):
    kwargs.update(_config.normConv_cfg)
    return _config.normConv_methods[_config.normConv](*args, **kwargs)
