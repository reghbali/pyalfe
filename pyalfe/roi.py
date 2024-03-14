import importlib

roi_dict = {
    'tissue_seg': {'sub_dir': None, 'measure': 'volume', 'type': 'derived'},
    'VentriclesDist': {'sub_dir': None, 'measure': 'distance', 'type': 'derived'},
    'template': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_BrainCerebellum.nii.gz'
        ),
    },
    'template_mask': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'aux',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_BrainCerebellumMask.nii.gz'
        ),
        'regions': {
            'Brain': [1],
        },
    },
    'lobes': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_Lobes.nii.gz'
        ),
        'regions': {
            'Frontal': [1],
            'Parietal': [2],
            'Occipital': [3],
            'Temporal': [4, 5, 6],
            'AnteriorTemporal': [4],
            'MiddleTemporal': [5],
            'PosteriorTemporal': [6],
            'Parietal_Occipital': [2, 3],
        },
    },
    'CorpusCallosum': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_CorpusCallosum.nii.gz'
        ),
        'regions': {
            'CorpusCallosum': [1, 2, 3, 4, 5],
            'CorpusCallosum_Rostrum': [1],
            'CorpusCallosum_Genu': [2],
            'CorpusCallosum_Body': [3],
            'CorpusCallosum_Isthmus': [4],
            'CorpusCallosum_Splenium': [5],
        },
    },
    'Tissue': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'oasis', 'T_template0_Tissue.nii.gz'
        ),
        'regions': {
            'CSF': [1],
            'Cortical Gray Matter': [2],
            'White Matter': [3],
            'Deep Gray Matter': [4],
            'Brain Stem': [5],
            'Cerebellum': [6],
        },
    },
}
