# OmniBoost

Boosting Throughput of Heterogeneous Embedded Devices under Multi-DNN Workload

![Methodology](docs/methodology.png)

### Abstract

Modern Deep Neural Networks (DNNs) exhibit profound efficiency and accuracy properties. This has introduced application workloads that comprise of multiple DNN applications, raising new challenges regarding workload distribution. Equipped with a diverse set of accelerators, newer embedded system present architectural heterogeneity, which current run-time controllers are unable to fully utilize. To enable high throughput in multi-DNN workloads, such a controller is ought to explore hundreds of thousands of possible solutions to exploit the underlying heterogeneity. In this paper, we propose OmniBoost, a lightweight and extensible multi-DNN manager for heterogeneous embedded devices. We leverage stochastic space exploration and we combine it with a highly accurate performance estimator to observe a x4.6 average throughput boost compared to other state-of-the-art methods. The evaluation was performed on the HiKey970 development board.

### References

* [DAC](www.example.com)

When using any of this project's source code, please cite:
```bibtex
@inproceedings{karatzas2023omniboost,
    title={OmniBoost: Boosting Throughput of Heterogeneous Embedded Devices under Multi-DNN Workload},
    author={Karatzas, Andreas and Anastasopoulos, Iraklis},
    booktitle={Proceedings of the 60th ACM/IEEE Design Automation Conference},
    pages={t.b.a.},
    year={2023}
}
```

### Installation 


For Windows:
```powershell
conda env create --file environment-windows.yml
conda activate omniboost
```

For Linux:
```powershell
conda env create --file environment-linux.yml
conda activate omniboost
```

If you either added or removed packages, then you can save a checkpoint of the `conda` environment by:

```powershell
conda env export --no-builds > environment-<os>.yml
```

The Linux distribution used for testing is Ubuntu 18.04.5. The Windows distribution used for testing is Windows 11 Pro, version 22H2.

### Optional

For a custom dataset, you need to train a standardization and a normalization model. This is to avoid any numerical pitfalls during the training of the throughput estimator. To do so, use the function `preprocessor` inside `common > preprocessor` as a template and pass the target `Y` vectors to it. This will generate two files, `StandardScaler.pkl` and `MinMaxScaler.pkl`, which you need to place under the `data > demo` directory (or  under any other custom directory). After that, you can uncomment the lines `275 - 276` in `envs/hikey.py` and lines `95 - 96, 99, 101, 106 - 111` in `envs/utils.py`.

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
