
import wandb
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../../')

import os
import time
import torch
import warnings
import datetime

from torch.utils.data import DataLoader

from common.terminal import colorstr
from lib.estimator.engine import Estimator
from lib.estimator.nvidia import cuda_check
from lib.estimator.logger import HardLogger
from lib.estimator.dataloader import CustomDataset
from lib.estimator.deterministic import set_deterministic
from lib.estimator.args import simulator_options, store_args
from lib.estimator.utils import info, confirm_paths
from lib.estimator.tracker import weight_histograms, TraceWrapper


def driver(args):
    warnings.filterwarnings("ignore")
    msg_logger = HardLogger(name=args.name, export_data_path=args.out_data_dir)
    
    for arg, value in sorted(vars(args).items()):
        msg_logger.logger.debug("Argument %s: %r", arg, value)
    
    confirm_paths(args=args)
    
    # NOTE: Do not tweak this line. If you change the path, then you shall also 
    #       reconfigure the args loader in the test script.
    config = store_args(
        filename=os.path.join(msg_logger.export_data_path, msg_logger.name, 'config.json'), 
        args=args
    )

    if args.use_tensorboard:
        # initialize tensorboard instance
        """To start a tensorboard instance, run the following command:
        
            >>> tensorboard --logdir=./experiments/ --host localhost --port 8888
        """
        writer = SummaryWriter(
            log_dir=msg_logger.log_dir, 
            comment=args.name if args.name else ""
        )
    
    if args.use_wandb:
        wandb.init(project=args.name, name=args.name, config=config)
    
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Device utilized: {colorstr(options=['red', 'underline'], string_args=list([device]))}.\n")
    
    if device == torch.device('cuda'):
        args.n_devices, cuda_arch = cuda_check()
        print(
            f"Found NVIDIA GPU of "
            f"{colorstr(options=['cyan'], string_args=list([cuda_arch]))} "
            f"Architecture.")
    
    if args.use_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
        set_deterministic(msg_logger=msg_logger, seed=args.seed)

    # dataloader training
    train_data = CustomDataset(
        root_dir=os.path.join(args.dataset_path, "train"), 
        use_gpu=True if 'cuda' in device.type.lower() and args.load_data else False
    )

    # dataloader validation
    val_data = CustomDataset(
        root_dir=os.path.join(args.dataset_path, "valid"), 
        use_gpu=True if 'cuda' in device.type.lower() and args.load_data else False
    )

    # dataloader training
    dataloader_train = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # dataloader validation
    dataloader_valid = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = Estimator(verbose=args.verbose, chkpt_interval=args.chkpt_interval,
                      dataset_id=args.dataset_id, logger=msg_logger, data_path=args.out_data_dir, 
                      loaded_data=True if 'cuda' in device.type.lower() and args.load_data else False,
                      lr=args.lr, weight_decay=args.weight_decay, num_devices=args.sim_num_devices,
                      clip_grad_norm=args.clip_grad_norm, device=device, load_checkpoint=args.resume, 
                      checkpoint_metric=args.elite_metric, auto_save=args.auto_save, 
                      overwrite_num_classes=args.overwrite_num_classes, epochs=args.epochs)

    if not args.graph and args.use_tensorboard:
        tb_model = TraceWrapper(model.dnn)
        tb_model.eval()
        writer.add_graph(
            model=tb_model,
            input_to_model=torch.rand(
                [args.sim_num_devices, args.sim_num_dnn, args.emb_dim]).unsqueeze(0).to(model.device),
            verbose=False,
            use_strict_trace=True
        )
        del tb_model

    msg_logger.print_training_message(
        model_id=args.model_id, dataset_id=args.dataset_id, epochs=args.epochs, device=args.device,
        elite_metric=args.elite_metric, epoch=model.epoch, resume=args.resume, auto_save=args.auto_save
    )

    if args.use_wandb:
        wandb.watch(model.dnn, criterion=model.criterion)

    msg_logger.logger.debug(f"Starting training model")
    start_time = time.time()
    for epoch in range(model.epoch, model.epochs):
        training_acc, training_loss = model.train(train_loader=dataloader_train, epoch=epoch)
        validation_acc, validation_loss = model.validate(valid_loader=dataloader_valid, epoch=epoch)
        model.checkpoint_metric_evaluation(mode="validation")

        if args.use_tensorboard:
            writer.add_scalar('lr/train', model.optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar('loss/train', training_loss, epoch)
            writer.add_scalar('acc/train', training_acc, epoch)
            writer.add_scalar('loss/valid', validation_loss, epoch)
            writer.add_scalar('acc/valid', validation_acc, epoch)
            # Visualize weight histograms
            # TODO: re-enable this once the project is published
            weight_histograms(writer, epoch, model)
        
        if args.use_wandb:
            wandb.log({'lr/train': model.optimizer.param_groups[0]["lr"]})
            wandb.log({'loss/train': training_loss})
            wandb.log({'acc/train': training_acc})
            wandb.log({'loss/valid': validation_loss})
            wandb.log({'acc/valid': validation_acc})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    msg_logger.logger.info(f"Training time - {total_time_str}")
    
    if args.use_tensorboard:
        # add experiment hyperparameters
        writer.add_hparams(hparam_dict=config, metric_dict={
            'Accuracy': model.best_acc,
            'Loss': model.best_loss
        })
        # close tensorboard writer
        writer.close()


if __name__ == "__main__":
    args = simulator_options().parse_args()
    if args.info:
        info()
    driver(args=args)
