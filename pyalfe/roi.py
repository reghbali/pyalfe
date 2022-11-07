import os

roi_dict = {
    'tissue_seg': {
        'sub_dir': None,
        'type': 'volume'
    },
    'VentriclesDist': {
        'sub_dir': None,
        'type': 'distance'
    },
    'template': {
        'sub_dir': 'TemplateReg',
        'type': 'volume',
        'source': os.path.join(os.path.dirname(__file__), 'templates', 'oasis',
                               'T_template0_BrainCerebellum.nii.gz'),
    },
    'template_mask': {
        'sub_dir': 'TemplateReg',
        'type': 'volume',
        'source': os.path.join(os.path.dirname(__file__), 'template', 'oasis',
                               'T_template0_BrainCerebellumMask.nii.gz')
    },
    'lobes': {
        'sub_dir': 'TemplateReg',
        'type': 'volume',
        'source': os.path.join(os.path.dirname(__file__),
                               'template', 'oasis', 'T_template0_Lobes.nii.gz')
    }
}