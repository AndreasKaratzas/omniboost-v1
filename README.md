# OmniBoost

Modern Deep Neural Networks (DNNs) exhibit profound efficiency and accuracy properties. This has introduced application workloads that comprise of multiple DNN applications, raising new challenges regarding workload distribution. Equipped with a diverse set of accelerators, newer embedded system present architectural heterogeneity, which current run-time controllers are unable to fully utilize. To enable high throughput in multi-DNN workloads, such a controller is ought to explore hundreds of thousands of possible solutions to exploit the underlying heterogeneity. In this paper, we propose OmniBoost, a lightweight and extensible multi-DNN manager for heterogeneous embedded devices. We leverage stochastic space exploration and we combine it with a highly accurate performance estimator to observe a x4.6 average throughput boost compared to other state-of-the-art methods. The evaluation was performed on the HiKey970 development board.

### Installation 

```powershell
conda env create --file environment.yml
conda activate omniboost
```

If you either added or removed packages, then you can save a checkpoint of the `conda` environment by:

```powershell
conda env export --no-builds > environment.yml
```

### Preliminaries

For a custom dataset, you need to train a standardization and a normalization model. This is to avoid any numerical pitfalls during the training of the throughput estimator. To do so, use the function `preprocessor` and pass the target (`Y`) vectors to it. This will generate two files, `StandardScaler.pkl` and `MinMaxScaler.pkl`, which you need to place under the `data > demo` directory (or  under any other custom directory). After that, you can uncomment the lines `275 - 276` in `envs/hikey.py` and lines `95 - 96, 99, 101, 106 - 111` in `envs/utils.py`.

### Usage

To first generate a fake dataset in order to test the code and understand the workflow, run:

```powershell
cd helper; python generator.py
```

This will generate 100 random samples under the `data > demo` directory, already split into training, validation and test sets.

Next, we train the throughput estimator CNN. Navigate to the parent project directory, and then run:

```powershell
cd lib/estimator; python train.py --model-id resnet9 --dataset-id 'demo' --use-deterministic-algorithms --dataset-path '../../data/demo/' --name demo --auto-save --info --out-data-dir '../../data/demo/experiments' --use-tensorboard --use-wandb
```

You can also test the estimator after training by running:

```powershell
python test.py --use-deterministic-algorithms --demo --resume '../../data/demo/experiments/<demo>/model/<model>.pth'
```

Finally, we are ready to use the MCTS algorithm to find the best configuration for our workload. Navigate to the parent project directory, and then run:

```powershell
cd lib/mcts; python main.py --seed 33 --workload 0 1 2 --use-deterministic-algorithms --demo --auto-set --resume '../../data/demo/experiments/<demo>/model/<model>.pth'
```

### Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{karatzas2023omniboost,
    title={OmniBoost: A Lightweight Multi-DNN Manager for Heterogeneous Embedded Devices},
    author={Karatzas, Andreas and Anastasopoulos, Iraklis},
    booktitle={Proceedings of the 60th ACM/IEEE Design Automation Conference},
    pages={1--6},
    year={2023}
}
```
