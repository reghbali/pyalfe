# PyALFE

PyALFE is an open-source python Automated Lesion and Feature Extraction pipeline.
It consumes a clinical brain MRI study consisting of various pulse sequences
such as T1, T1 post contrast, FLAIR, T2, ADC, and CBF. It performs
pre-processing, inter-seq registration, template registration, segmentation,
and quantification to generate a set of numerical human-interpretable features
corresponding to the lesions found in the MRI scan.
