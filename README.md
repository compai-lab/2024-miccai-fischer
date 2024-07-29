# Code for Progressive Growing of Patch Size Curriculum implemented into the nnUNet framework.

![grafik](https://github.com/user-attachments/assets/840d1c17-de81-49f8-bd51-3644378a3131)

![grafik](https://github.com/user-attachments/assets/96cd3e81-cb11-4a3d-acd4-fb8ef59669b9)



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

