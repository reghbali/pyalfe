import importlib
import os

roi_dict = {
    'tissue_seg': {
        'sub_dir': None,
        'measure': 'volume',
        'type': 'derived'
    },
    'VentriclesDist': {
        'sub_dir': None,
        'measure': 'distance',
        'type': 'derived'
    },
    'template': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'registered',
        'source':  importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_BrainCerebellum.nii.gz'),
    },
    'template_mask': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'aux',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_BrainCerebellumMask.nii.gz')
    },
    'lobes': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'registered',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_Lobes.nii.gz')
    }
}