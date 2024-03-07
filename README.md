# Code for Progressive Growing of Patch Size Curriculum implemented into the nnUNet framework.
Training code for PGPS/PGPS+/RPSS can be found in directory:

- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_PGPS.py
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_PGPS+.py
- ./training/nnUNetTrainer/variants/sampling/nnUNetTrainer_RPSS.py

For using the code, please install nnUNet from https://github.com/MIC-DKFZ/nnUNet according to their documentation.
After that you can copy the training code plus:
- ./utilities/plans_handling/plans_handler.py

Do training as standard nnUNet as specified here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
The only difference is to specifiy the PGPS/PGPS+/RPSS trainer:
```
# for PGPS
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_PGPS
# for PGPS+
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_PGPS+
# for RPSS
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_RPSS
```
