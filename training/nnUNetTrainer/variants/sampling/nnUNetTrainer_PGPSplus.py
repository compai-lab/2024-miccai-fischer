import torch 
from torch import autocast, nn 
from torch import distributed as dist 
from torch.cuda import device_count 
from torch.cuda.amp import GradScaler 
from torch.nn.parallel import DistributedDataParallel as DDP 
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer 
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerBN import nnUNetTrainerBN 
import time 
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

# track CO2 emissions
# from codecarbon import EmissionsTracker

class nnUNetTrainer_PGPSplus(nnUNetTrainer):

    def plan_batch_size(self):
        """
        compute maximal possible batch size for each patch size stage, such that the number of input voxels per input tensor increases monotonically over training
        """
        current_patch_size = self.min_patch_size
        patch_sizes = []
        patch_sizes.append(current_patch_size)
        new_min_patch_size = current_patch_size
        i = 1
        while True:
            while (new_min_patch_size == current_patch_size).all():
                if (self.original_patch_size == current_patch_size).all():
                    break
                idx = i % 3
                add = np.array([0, 0, 0])
                add[idx] = 1
                new_min_patch_size = new_min_patch_size + add * np.array(
                    [(2 ** self.num_pool_per_axis[0]), (2 ** self.num_pool_per_axis[1]),
                     (2 ** self.num_pool_per_axis[2])])
                new_min_patch_size = np.where(new_min_patch_size > self.original_patch_size, self.original_patch_size,
                                              new_min_patch_size)
                i = i + 1
            current_patch_size = new_min_patch_size
            patch_sizes.append(current_patch_size)
            if (self.original_patch_size == current_patch_size).all():
                break

        print("All patchsizes:")
        print(patch_sizes)
        batch_sizes = [2]
        for i in range(len(patch_sizes) - 2, -1, -1):
            current_patch_size = patch_sizes[i]
            higher_patch_size = patch_sizes[i + 1]
            ratio = ((higher_patch_size[0] * higher_patch_size[1] * higher_patch_size[2]) / (
                        current_patch_size[0] * current_patch_size[1] * current_patch_size[2]))
            batch_size = int(ratio * batch_sizes[-1])
            batch_sizes.append(batch_size)

        for i in range(len(batch_sizes)):
            batch_sizes[i] = max(batch_sizes[i], self.original_batch_size)
        batch_sizes = batch_sizes[::-1]
        batch_sizes.append(self.original_batch_size)
        print("All batchsizes: ")
        print(batch_sizes)
        print(len(batch_sizes))
        self.batch_sizes = batch_sizes

    def get_max_bs(self):
        max_bs = self.batch_sizes[self.stage]
        return max_bs

    def run_training(self):
        """
        training loop for Progressive Growing Of Patch Size:
        1. We start training on minimal possible patch size (depends on network architecture/num of pooling operations)
        2. We increase the patch size linearly during training using the minimal possible patch size steps (depends on network architecture/num of pooling operations)
        3. We use the final patch size as inference patch size

        Minimal Patch Size: minimal patch size that does result in 1x2x1 vector in bottleneck (depends on network architecture/num of pooling operations)
        Patch Size Steps: we take the minimal possible patch size update (only even integer shapes in all network stages are allowed in network processing)
        Maximal Patch Size: we take the same patch size as standard nnUNet planning (nnUNet maximizes the patch size that can fit with a batch size of 2 into a GPU memory of 11GB)
        """

        self.on_train_start() 
        self.initialized = False
        self.original_patch_size = self.configuration_manager.patch_size 
        self.original_batch_size = self.configuration_manager.batch_size
        self.print_to_log_file('######################### Progressive Growing of Patchsize Training Config #########################') 
        self.print_to_log_file('Original patch size: ' + str(self.configuration_manager.patch_size))
        self.print_to_log_file('Original batch size: ' + str(self.configuration_manager.batch_size))  
        num_pool_per_axis = self.configuration_manager.num_pool_per_axis 
        self.print_to_log_file('num_pool_per_axis: ' + str(num_pool_per_axis)) 
        self.min_patch_size = np.array([2**(num_pool_per_axis[0]+1), 2**(num_pool_per_axis[1]), 2**(num_pool_per_axis[2])])
        current_patch_size = self.min_patch_size 
        self.print_to_log_file('Minimal possible patch size: ' + str(self.min_patch_size)) 
        self.print_to_log_file('######################### Progressive Growing of Patchsize Training Config #########################')
        self.num_pool_per_axis = num_pool_per_axis
        self.oversample_foreground_percent = 0.5
        self.stage = 0
        self.plan_batch_size()
        # start tracking C02 emissions
        # tracker = EmissionsTracker(output_dir=self.output_folder)
        # tracker.start()
        for epoch in range(self.current_epoch, self.num_epochs):
            if not self.initialized:
                # initialize training with smallest possible patchsize (depends on num of pooling operations) 
                self.patch_size = self.min_patch_size
                self.print_to_log_file('Current patchsize : ' + str(self.patch_size))
                self.configuration_manager.set_patch_size(self.patch_size)
                self.batch_size = self.get_max_bs()
                self.configuration_manager.set_batch_size(self.batch_size)
                self.print_to_log_file('Current batchsize : ' + str(self.batch_size)) 
                self.dataloader_train_cur_patch_size, _ = self.get_dataloaders()
                # Max Patch Training
                self.patch_size = self.original_patch_size 
                self.configuration_manager.set_patch_size(self.patch_size) 
                self.batch_size = 2
                self.configuration_manager.set_batch_size(self.batch_size)
                self.dataloader_train_max_patch_size, self.dataloader_val_max_patch_size = self.get_dataloaders() 
                self.initialized = True

                # counter for increasing patchsize, we wanna start with bottleneck axis of length=1 
                i = 1
                num_stages = np.ceil((self.original_patch_size[0] - self.min_patch_size[0])/ (2**num_pool_per_axis[0])) + np.ceil((self.original_patch_size[1] - self.min_patch_size[1])/ (2**num_pool_per_axis[1])) + np.ceil((self.original_patch_size[2] - self.min_patch_size[2])/ (2**num_pool_per_axis[2])) + 1
                self.print_to_log_file('Number of different patchsize phases : ' + str(num_stages))
                self.patch_size = self.min_patch_size

            elif ((self.current_epoch % int(self.num_epochs / num_stages)) == 0):
                # save checkpoint after finish patchsize phase
                chk_name = 'checkpoint_' +  str(self.patch_size[0]) + 'x' + str(self.patch_size[1]) + 'x' + str(self.patch_size[2]) + '.pth' 
                self.save_checkpoint(join(self.output_folder, chk_name))

                self.stage = self.stage + 1
                # increase patch size in minimal possible steps, for each patchsize-phase same number of epochs 
                new_current_patch_size = current_patch_size
                while (new_current_patch_size == current_patch_size).all():
                    if (new_current_patch_size == self.original_patch_size).all():
                        break 
                    idx = i % 3 
                    add = np.array([0, 0, 0]) 
                    add[idx]  = 1 
                    new_current_patch_size = new_current_patch_size + add * np.array([2**num_pool_per_axis[0], 2**num_pool_per_axis[1], 2**num_pool_per_axis[2]]) 
                    new_current_patch_size = np.where(new_current_patch_size > self.original_patch_size, self.original_patch_size, new_current_patch_size) 
                    i = i + 1 

                self.patch_size = new_current_patch_size
                current_patch_size = self.patch_size

                self.configuration_manager.set_patch_size(self.patch_size) 
                self.print_to_log_file('New Patch Size : ' + str(self.patch_size))
                self.batch_size = self.get_max_bs()
                self.configuration_manager.set_batch_size(self.batch_size)
                self.print_to_log_file('New Batch Size : ' + str(self.batch_size))  
                self.dataloader_train_cur_patch_size, _ = self.get_dataloaders()

            # training process: train on current patchsize
            # tracker.start_task(str('Task' +  str(self.patch_size[0]) + 'x' + str(self.patch_size[1]) + 'x' + str(self.patch_size[2])))
            self.on_epoch_start() 
            self.on_train_epoch_start() 
            train_outputs = [] 
            for batch_id in range(int(self.num_iterations_per_epoch)): 
                train_outputs.append(self.train_step(next(self.dataloader_train_cur_patch_size)))
            self.on_train_epoch_end(train_outputs)            

            # validation process: evaluate current weights on final patchsize
            with torch.no_grad(): 
                self.on_validation_epoch_start() 
                val_outputs = [] 
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val_max_patch_size))) 
            self.on_validation_epoch_end(val_outputs) 
            self.on_epoch_end()
            # emissions = tracker.stop_task()
            # print(emissions)

        # finish tracking
        # emissions = tracker.stop()


        print(f"Emissions : {1000 * emissions} g CO₂") 
        for task_name, task in tracker._tasks.items():
            self.print_to_log_file( f"Emissions : {1000 * task.emissions_data.emissions} g CO₂ for task {task_name}")


        # test model on test split
        self.configuration_manager.set_patch_size(self.original_patch_size) 
        self.configuration_manager.set_batch_size(self.original_batch_size) 
        self.oversample_foreground_percent = 0.33
        self.on_train_end()


    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')): 
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device) 
        self.num_iterations_per_epoch = 250



