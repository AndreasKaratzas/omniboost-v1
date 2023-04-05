
import torch
import numpy as np
import torch.nn as nn

from lib.estimator.resnet import factory as resnet_factory


class Tester(nn.Module):
    def __init__(
        self,
        num_devices: int, 
        device: torch.device,
        verbose: bool = False,
        load_checkpoint: str = None,
        overwrite_num_classes: int = None
    ):
        super(Tester, self).__init__()
        
        self.dnn = self._model_factory(
            num_devices=num_devices, 
            overwrite_num_classes=overwrite_num_classes)

        if verbose:
            print(self.dnn)

        self.device = device
        
        self.load(checkpoint=load_checkpoint)
        self.dnn = self.dnn.to(self.device)

        # Log number of parameters
        print(f"Number of parameters found in module: {self._count_vars()}")
    
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

    def _count_vars(self) -> int:
        """Counts trainable parameters of nn.Module.

        Returns
        -------
        int
            Number of total trainable parameters found in the model.
        """
        return sum([np.prod(p.shape) for p in self.parameters()])
    
    def test(self, mapping: np.ndarray):
        """Invoke the model on a single test sample.
        
        Parameters
        ----------
        mapping : np.ndarray
            The single input vector.
        
        Returns
        -------
        torch.Tensor
            Prediction (workload) on the custom input.
        """
        # Switch to evaluation mode
        self.dnn.eval()

        with torch.no_grad():

            # map data onto device
            mapping = torch.from_numpy(mapping).type(
                torch.FloatTensor).to(self.device)

            if len(mapping.shape) == 3:
                mapping = mapping.unsqueeze(0)

            # compute output from model
            output = self.dnn(mapping)

            # detach model prediction
            output = output.detach().cpu().numpy()

        return output

    def load(self, checkpoint: str = None):
        """Load a checkpoint.
        
        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint to load.
        """

        checkpoint = torch.load(checkpoint)
        self.epoch = checkpoint['epoch']
        self.dnn.load_state_dict(checkpoint['model_state_dict'])
        self.best_acc = checkpoint['acc']
        self.best_loss = checkpoint['loss']
