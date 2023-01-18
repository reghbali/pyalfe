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
        'type': 'template',
        'source':  importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_BrainCerebellum.nii.gz'),
        'regions': {
            'Brain': [1],
         }
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
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_Lobes.nii.gz'),
        'regions': {
            'Frontal': [1],
            'Parietal': [2],
            'Occipital': [3],
            'Temporal': [4,5,6],
            'AnteriorTemporal': [4],
            'MiddleTemporal': [5],
            'PosteriorTemporal': [6],
            'Parietal_Occipital': [2,3]
         }
    }
}