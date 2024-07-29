# Code for Progressive Growing of Patch Size Curriculum implemented into the nnUNet framework.

This is the implementation from our MICCAI paper "Progressive Growing of Patch Size: Resource-Efficient Curriculum Learning for Dense Prediction Tasks". Abstract. In this work, we introduce Progressive Growing of Patch
Size, a resource-efficient implicit curriculum learning approach for dense prediction tasks. Our curriculum approach is defined by growing the patch size during model training, which gradually increases the task’s difficulty. We integrated our curriculum into the nnU-Net framework and evaluated the methodology on all 10 tasks of the Medical Segmentation Decathlon. With our approach, we are able to substantially reduce runtime, computational costs, and CO2 emissions of network training compared to classical constant patch size training. In our experiments, the curriculum approach resulted in improved convergence. 





**Curriculum Learning Trainer Script Implemented in nnunetv2=2.0.0.**

**The configuration manager of nnUNetv2 was changed in newer versions and is not yet compatible with this nnUnet trainer script!!!**

**Please use nnunetv2=2.0.0 for this repo!**


Training code for PGPS/PGPSplus/RPSS can be found in directory:

- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_PGPS.py
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_PGPSplus.py
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_RPSS.py

For using the code, please install nnUNet from https://github.com/MIC-DKFZ/nnUNet according to their documentation.
After that you can copy the training code plus a plans handler file from THIS github repo into the python site-package nnUNetv2 installation folder:
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_PGPS.py
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_PGPSplus.py
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_RPSS.py
- ./utilities/plans_handling/plans_handler.py

Do training as standard nnUNet as specified here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
The only difference is to specifiy the PGPS/PGPS+/RPSS trainer:
```
# for PGPS
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_PGPS
# for PGPS+
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_PGPSplus
# for RPSS
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_RPSS
```
**comment codecarbon parts if you do not want to track C02-equivalents!**
\\
For more information about codecarbon check their website: https://mlco2.github.io/codecarbon/installation.html

