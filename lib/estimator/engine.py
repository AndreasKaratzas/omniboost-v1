
import os
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.estimator.resnet import factory as resnet_factory
from lib.estimator.metric import MetricLogger, SmoothedValue, CustomAverageValueMeter


class Estimator(nn.Module):
    def __init__(
        self, 
        logger,
        data_path: str,
        dataset_id: str,
        num_devices: int, 
        weight_decay: float,
        lr: float,
        epochs: int,
        device: torch.device,
        auto_save: bool,
        chkpt_interval: int,
        clip_grad_norm: float = None,
        checkpoint_metric: str = "acc",
        overwrite_num_classes: int = None,
        verbose: bool = False,
        load_checkpoint: bool = False,
        loaded_data: bool = False
    ):
        super(Estimator, self).__init__()

        self.loaded_data = loaded_data
        self.chkpt_cntr = 0
        self.epoch = 0
        self.epochs = epochs
        self.auto_save = auto_save
        self.chkpt_interval = chkpt_interval
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.overwrite_num_classes = overwrite_num_classes

        self.dnn = self._model_factory(
            num_devices=num_devices, 
            overwrite_num_classes=overwrite_num_classes)

        if verbose:
            print(self.dnn)
        
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_metric = checkpoint_metric
        
        if load_checkpoint:
            self.load()
        self.dnn = self.dnn.to(self.device)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Log number of parameters
        print(f"Number of parameters found in module: {self._count_vars()}")

        # Declare logger
        self.logger = logger

        # Initialize elite metrics
        self.best_acc = 0.0
        self.best_loss = np.inf

    def _count_vars(self) -> int:
        """Counts trainable parameters of nn.Module.

        Returns
        -------
        int
            Number of total trainable parameters found in the model.
        """
        return sum([np.prod(p.shape) for p in self.parameters()])
    
    def _model_factory(self, num_devices: int = 3, overwrite_num_classes: int = None) -> nn.Module:
        """Builds a DNN model to be used as a __Estimator__. The instance
        is compiled using the local `resnet` modules.

        Parameters
        ----------
        num_devices : int, optional (default=3)
            Number of devices found in the heterogeneous platform.
        overwrite_num_classes : int, optional (default=None)
            Overwrites the number of classes in the model. If None, the number of classes is set to `num_devices`.

        Returns
        -------
        nn.Module
            The __Estimator__ model.
        """
        return resnet_factory(in_channels=num_devices, n_classes=overwrite_num_classes if overwrite_num_classes else num_devices)
        
    def _calculate_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate accuracy of predictions.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions.
        y_true : torch.Tensor
            Ground truth.

        Returns
        -------
        torch.Tensor
            Accuracy.
        """
        return 1 - torch.mean(torch.abs(y_pred - y_true)).detach().cpu().numpy()

    def _reserved_cuda_memory(self) -> int:
        """Calculate reserved cuda memory.

        Returns
        -------
        int
            Reserved cuda memory.
        int
            Scale factor for reserved cuda memory.
        """
        
        scales = ['B', 'K', 'M', 'G']
        mem_size = torch.cuda.memory_reserved()
        for scale in scales: 
            if mem_size > 1024:
                mem_size /= 1024
            else:
                return int(round(mem_size)), scale
        return int(round(mem_size)), scale

    def train(self, train_loader: DataLoader, epoch: int = -1):
        """Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        epoch : int, optional
            Epoch number. If -1, the current epoch is used.
        
        Returns
        -------
        float
            Accuracy of the model on the training set.
        float
            Loss of the model on the training set.
        """

        self.epoch = epoch if self.epoch < epoch else self.epoch

        losses = {'overall_loss': CustomAverageValueMeter()}
        model_acc = CustomAverageValueMeter()

        total_samples = len(train_loader.sampler)
        batch_size = train_loader.batch_size
        total_steps = total_samples / batch_size

        terminal_logger = MetricLogger(
            delimiter=" ", log_dir=self.logger.log_dir)
        terminal_logger.add_meter('lr', SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
        steps_completed = 0

        # Switch to train mode
        self.dnn.train()

        print(
            f"\n\n{'':>10}{'Epoch':>10}{'subset':>10}{'gpu_mem':>10}{'loss':>10}{'acc':>10}")
        with tqdm(total=len(train_loader), bar_format='{l_bar}{bar:25}{r_bar}{bar:-25b}') as pbar:
            for images, targets in terminal_logger.log_every(train_loader, epoch):

                if not self.loaded_data:
                    images, targets = images.to(
                        self.device), targets.to(self.device)

                output = self.dnn(images)
                loss = self.criterion(output, targets)

                # Record loss
                losses['overall_loss'].add(loss.item())

                # Record accuracy
                model_acc.add(self._calculate_accuracy(output, targets))

                # Compute the gradient and do optimizer step
                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.dnn.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                terminal_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                # measure elapsed time
                steps_completed = steps_completed + 1

                # get cuda memory usage
                if 'cuda' in self.device.type.lower():
                    mem_size, scale = self._reserved_cuda_memory()
                else:
                    mem_size, scale = 0, 'B'
                
                pbar.set_description(('%10s' + '%10s' + '%10s' + '%10s' + '%10.3g' + '%10.3g') % (
                    f'',
                    f'{epoch + 1}/{self.epochs}',
                    f'{steps_completed}/{int(np.ceil(total_steps))}',
                    f'{mem_size:.3g} ' + scale,
                    round(losses['overall_loss'].value()[0], 3), round(model_acc.value()[0], 3)))

                pbar.update(1)

            pbar.close()

        self.training_loss = losses['overall_loss'].value()[0]
        self.training_acc = model_acc.value()[0]

        self.logger.log_stats(
            acc=model_acc.value()[0],
            loss=losses['overall_loss'].value()[0],
            epoch=epoch, desc="Training"
        )

        return model_acc.value()[0], losses['overall_loss'].value()[0]

    def validate(self, valid_loader: DataLoader, epoch: int = -1):
        """Validate the model on the validation set.
        
        Parameters
        ----------
        valid_loader : DataLoader
            DataLoader for the validation set.
        epoch : int, optional
            Epoch number. Defaults to -1.
        
        Returns
        -------
        float
            Accuracy of the model on the validation set.
        float
            Loss of the model on the validation set.
        """

        losses = {'overall_loss': CustomAverageValueMeter()}
        model_acc = CustomAverageValueMeter()

        batch_time = CustomAverageValueMeter()
        total_samples = len(valid_loader.sampler)
        batch_size = valid_loader.batch_size

        terminal_logger = MetricLogger(
            delimiter=" ", log_dir=self.logger.log_dir)

        total_steps = total_samples / batch_size
        steps_completed = 0

        # Switch to evaluation mode
        self.dnn.eval()

        with torch.no_grad():
            with tqdm(total=len(valid_loader), bar_format='{l_bar}{bar:25}{r_bar}{bar:-25b}') as pbar:
                for images, targets in terminal_logger.log_every(valid_loader, epoch):
                    
                    if not self.loaded_data:
                        images, targets = images.to(
                            self.device), targets.to(self.device)
                    end = time.time()

                    # compute output from model
                    output = self.dnn(images)

                    # measure elapsed time
                    batch_time.add(time.time() - end)

                    # compute loss
                    loss = self.criterion(output, targets)

                    # Record accuracy
                    model_acc.add(self._calculate_accuracy(output, targets))

                    # measure accuracy and record loss
                    losses['overall_loss'].add(loss.item())
                    
                    steps_completed = steps_completed + 1

                    # get cuda memory usage
                    if 'cuda' in self.device.type.lower():
                        mem_size, scale = self._reserved_cuda_memory()
                    else:
                        mem_size, scale = 0, 'B'

                    pbar.set_description(('%10s' + '%10s' + '%10s' + '%10s' + '%10.3g' + '%10.3g') % (
                        f'{" ".ljust(10)}',
                        f'{" ".ljust(10)}',
                        f'{steps_completed}/{int(np.ceil(total_steps))}',
                        f'{mem_size:.3g} ' + scale,
                        round(losses['overall_loss'].value()[0], 3), round(model_acc.value()[0], 3)))

                    pbar.update(1)

                pbar.close()
        
        self.validation_loss = losses['overall_loss'].value()[0]
        self.validation_acc = model_acc.value()[0]

        self.logger.log_stats(
            acc=model_acc.value()[0],
            loss=losses['overall_loss'].value()[0],
            epoch=epoch, desc="Validating"
        )

        return model_acc.value()[0], losses['overall_loss'].value()[0]

    def load(self, checkpoint: str = None):
        """Load a checkpoint.
        
        Parameters
        ----------
        checkpoint : str, optional
            Path to the checkpoint to load. Defaults to None.
    
        """

        if checkpoint is None:
            checkpoint = os.path.join(
                self.data_path, self.dataset_id, "estimator.pth")

        checkpoint = torch.load(checkpoint)
        self.epoch = checkpoint['epoch']
        self.dnn.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_acc = checkpoint['acc']
        self.best_loss = checkpoint['loss']

    def store(self):
        """Store the current instance state in a checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.dnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': self.best_acc,
            'loss': self.best_loss,
        }, os.path.join(
            self.logger.model_dir,
            f"epoch_{self.epoch:05d}-" + f"acc_{self.best_acc:07.3f}-" +
            f"loss_{self.best_loss:07.3f}.pth"
        ))

    def checkpoint_metric_evaluation(
        self,
        mode: str = "validation"
    ) -> bool:
        """Checkpoint the model if the metric is better than the previous best.

        Parameters
        ----------
        acc : float, optional
            Accuracy of the model. Defaults to None.
        loss : float, optional
            Loss of the model. Defaults to None.
        
        Returns
        -------
        bool
            Whether the instance was exported.
        """
        acc = self.validation_acc if mode == "validation" else self.training_acc
        loss = self.validation_loss if mode == "validation" else self.training_loss
        
        self.chkpt_cntr = self.chkpt_cntr + 1

        flag = False
        if acc > self.best_acc:
            self.best_acc = acc
            flag = True
        if loss < self.best_loss:
            self.best_loss = loss
            flag = True

        if flag and self.checkpoint_metric == "acc" and self.auto_save:
            self.store()
            return True
        elif flag and self.checkpoint_metric == "loss" and self.auto_save:
            self.store()
            return True
        else:
            if self.chkpt_cntr % self.chkpt_interval == 0 and not self.auto_save:
                self.store()
                return True
        return False
