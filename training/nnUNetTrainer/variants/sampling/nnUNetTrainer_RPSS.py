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

class nnUNetTrainer_RPSS(nnUNetTrainer):

    def run_training(self):
        """
        training loop for Random Patch Size Sampling:
        1. For each training iteration sample a patch size from the pool of possible patch sizes
        2. Training iteration on that patch size
        3. We use the maximal patch size as inference patch size

        Possible Patch Sizes: All patch sizes used for PGPS/PGPS+
        Minimal Satch Size: patch size that does result in 1x2x1 vector in bottleneck (depends on network architecture/num of pooling operations)
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
        self.train_dataloader_list = []
        # replicate standard nnUNet setting (batchsize=2 -> oversampling=0.5)
        self.oversample_foreground_percent = 0.5 
        
        # start tracking C02 emissions
        # tracker = EmissionsTracker(output_dir=self.output_folder)
        # tracker.start()
        for epoch in range(self.current_epoch, self.num_epochs):
            if not self.initialized:
                # Max Patch Training
                self.patch_size = self.original_patch_size 
                self.configuration_manager.set_patch_size(self.patch_size) 
                dataloader_train_max_patch_size, self.dataloader_val_max_patch_size = self.get_dataloaders() 
                self.initialized = True
                self.train_dataloader_list.append(dataloader_train_max_patch_size)
                # counter for increasing patchsize, we wanna start with bottleneck axis of length=1 
                i = 1
                num_stages = np.ceil((self.original_patch_size[0] - self.min_patch_size[0])/ (2**num_pool_per_axis[0])) + np.ceil((self.original_patch_size[1] - self.min_patch_size[1])/ (2**num_pool_per_axis[1])) + np.ceil((self.original_patch_size[2] - self.min_patch_size[2])/ (2**num_pool_per_axis[2])) + 1
                self.print_to_log_file('Number of different patchsize phases : ' + str(num_stages))
                self.patch_size = self.min_patch_size
                self.configuration_manager.set_patch_size(self.patch_size)
                dataloader_train_cur_patch_size, _ = self.get_dataloaders()
                self.train_dataloader_list.append(dataloader_train_cur_patch_size)
                self.print_to_log_file('create dataloader with patch size: ' + str(self.patch_size))


                # increase patch size in minimal possible steps, for each patchsize-phase same number of epochs 
                new_current_patch_size = current_patch_size
                while not (new_current_patch_size == self.original_patch_size).all():
                    idx = i % 3 
                    add = np.array([0, 0, 0]) 
                    add[idx]  = 1 
                    new_current_patch_size = new_current_patch_size + add * np.array([2**num_pool_per_axis[0], 2**num_pool_per_axis[1], 2**num_pool_per_axis[2]]) 
                    new_current_patch_size = np.where(new_current_patch_size > self.original_patch_size, self.original_patch_size, new_current_patch_size) 
                    i = i + 1
                    if (new_current_patch_size != current_patch_size).any():
                        # add new dataloader
                        self.print_to_log_file('create dataloader with patch size: ' + str(new_current_patch_size))
                        self.patch_size = new_current_patch_size
                        self.configuration_manager.set_patch_size(self.patch_size) 
                        dataloader_train_cur_patch_size, _ = self.get_dataloaders()
                        self.train_dataloader_list.append(dataloader_train_cur_patch_size)
                        current_patch_size = new_current_patch_size

            self.print_to_log_file('len of dataloader list: ' + str(len(self.train_dataloader_list)))
            # training process: train on current patchsize
            # tracker.start_task(str('Epoch:' + str(self.current_epoch)))
            self.on_epoch_start() 
            self.on_train_epoch_start() 
            train_outputs = []
            for batch_id in range(int(self.num_iterations_per_epoch)):
                loader_idx = np.random.randint(0, num_stages) 
                train_data = next(self.train_dataloader_list[loader_idx])
                train_outputs.append(self.train_step(train_data))
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


        # print(f"Emissions : {1000 * emissions} g CO₂") 
        # for task_name, task in tracker._tasks.items():
        #     self.print_to_log_file( f"Emissions : {1000 * task.emissions_data.emissions} g CO₂ for task {task_name}")


        # test model on test split
        self.configuration_manager.set_patch_size(self.original_patch_size) 
        self.configuration_manager.set_batch_size(self.original_batch_size) 
        self.oversample_foreground_percent = 0.33
        self.on_train_end()


    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')): 
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
