# Source Code for the Experiments conducted as part of the Master Thesis: A Solution for Decentralized Federated Multi-Task Learning
**Nicolas Kohler, 2024, University of Zurich, Department of Informatics (IFI), Communication Systems Group (CSG)**

The provided source code simulates a Federated Learning environment on one machine, to test the proposed Federated Multi-Task framework. The aim is to investigate how the framework handles different client heterogeneity types and not to create a proper federated environment. For this, a total of six experiments are run on two different dataset, CIFAR-10 and CelebA. Three of those focus on class label heterogenity, while the other three investigate task heterogenity. Eventually, these six experiments required the training of about 100 models.

All trained clients, plots, tables and used configs are accessible here: [https://drive.google.com/drive/folders/1Au6neZziuD0q4_pd3T8qO6-kMceEvVpK?usp=sharing](https://drive.google.com/drive/folders/1Au6neZziuD0q4_pd3T8qO6-kMceEvVpK?usp=sharing)

## Acknowledgement
Two main inspirations for the framework were FedPer and FedHCA2:

- **FedHCA2**: Lu, Y., Huang, S., Yang, Y., Sirejiding, S., Ding, Y., & Lu, H. (2023). Towards Hetero-Client Federated Multi-Task Learning. arXiv preprint arXiv:2311.13250.

- **FedPer**: Arivazhagan, M. G., Aggarwal, V., Singh, A. K., & Choudhary, S. (2019). Federated learning with personalization layers. arXiv preprint arXiv:1912.00818.

## Installation
### Required Steps
1) Make sure Conda is installed: [https://anaconda.org/anaconda/conda](https://anaconda.org/anaconda/conda)
2) Create the conda environment. Specifications are located in the environment.yml file.
    ```
    conda env create -f environment.yml
    ```
3) Activate Environment
    ```
    conda activate asfdfmtl
    ```
4) If you want to train on a **GPU** install CUDA support. For this to work you need the appropriate hardware and CUDA toolkit. More information can be found here: [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). Note that training on a GPU is highly recommended - Code was **not** tested on CPU!
    ```
    conda install pytorch-cuda=12.1 -c nvidia
    ```
5) If installed, CUDA can be tested with the following command.
    ```
    nvidia-smi
    ```

### Testing the Installation.
To test whether the installation was succesful one can run the tiny_cifar-10 configs. These experiments only include a few epochs and aggregation rounds. As such, the required computation time is limited. However, they do not provide any insights and are solely for testing purposes. Please see the tiny_cifar-10 configs for additional comments on the parameter configuration. Once the federation has been run, the "cross_configuration_plots_and_tables" results can be compared with the figures in (figures/installation_check). If the results match, full reproducibility is likely in the other experiments.

Note that upon running the code snippet a data folder, checkpoint folder and results folder will be automatically created.

#### Tiny Cifar-10
- Run the tiny federation.
    ```
    python src/run.py --configs_folder configs/tiny_cifar-10 --configs tiny_cifar-10_none_none.yml tiny_cifar-10_fedper_none.yml tiny_cifar-10_fedper_fedper.yml tiny_cifar-10_fedper_hca.yml 
    ```
- Visualize the tiny federation.
    ```
    python src/plot.py --dataset cifar10 --folder tiny_cifar-10 --cross_configuration_plots_and_tables --vag --vag_inputs Tiny_Cifar-10_None_None Tiny_Cifar-10_FedPer_None Tiny_Cifar-10_FedPer_FedPer Tiny_Cifar-10_FedPer_HCA
    ```
- Running this tiny experimental setup on a GeForce 3080 only takes a couple of minutes and is great to quickly test whether or not the installation was succesfull. As such, it does **NOT** represent the actual results. The actual results of the thesis are located here: [https://drive.google.com/drive/folders/1Au6neZziuD0q4_pd3T8qO6-kMceEvVpK?usp=sharing](https://drive.google.com/drive/folders/1Au6neZziuD0q4_pd3T8qO6-kMceEvVpK?usp=sharing)

## Reproducing the Experiments
All the experiments discussed in the thesis made use of the provided configs and were trained on a GeForce GRTX 3080. More information about reproducibility can be found in the "Testing the Installation" section above and on PyTorch: 
[https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)

Two datasets were used: CIFAR-10 and CelebA. These datasets are automatically downloaded upon code execution.
### CIFAR-10 - Class Label Heterogenity
The CIFAR-10 dataset is used to conduct single label image classification tasks. To model class label heterogenity, two groups of clients were defined.
One group focuses on classifying animals, while the other is tasked with classifying objects. On this dataset, 3 different experiments, that in total required the training of 10 federations (resulting in 60 models), were conducted. For details about the specific setups, please see the respective config files.
#### CIFAR-10 with Different Aggregation Schemes
This experiment compares the proposed framework with pure local training, backbone averaging within the taskgroup, as well as with backbone averaging within and across task groups. (Approx 3h on GeForce 3080)
- Run the federation:
    ```
    python src/run.py --configs_folder configs/cifar-10 --configs cifar-10_none_none.yml cifar-10_fedper_none.yml cifar-10_fedper_fedper.yml cifar-10_fedper_hca.yml 
    ```
- Visualize the results:
    ```
    python src/plot.py --dataset cifar10 --folder cifar-10 --cross_configuration_plots_and_tables --vag --vag_inputs  CIFAR-10_None_None CIFAR-10_Fedper_None CIFAR-10_Fedper_Fedper CIFAR-10_Fedper_HCA
    ```

### CIFAR-10 With Varying Backbone Layers
This experiment runs various configurations of the proposed framework, with varying the backbone/head split. (Approx 2h on GeForce 3080)
- Run the federation:
    ```
    python src/run.py --configs_folder configs/cifar-10_vbl --configs cifar-10_fedper_hca_bl-0.yml cifar-10_fedper_hca_bl-1.yml cifar-10_fedper_hca_bl-2.yml
    ```
- Visualize the results:
    ```
    python src/plot.py --dataset cifar10 --folder cifar-10_vbl --cross_configuration_plots_and_tables --vbl --vbl_inputs CIFAR-10_FedPer_HCA_BL-0 CIFAR-10_Fedper_HCA_BL-1 CIFAR-10_FedPer_HCA_BL-2
    ```

### CIFAR-10 With Varying Epochs per Round
This experiment runs various configurations of the proposed framework, with varying amounts of epochs between aggregation rounds. (Approx 2h on GeForce 3080)
- Run the federation:
    ```
    python src/run.py --configs_folder configs/cifar-10_vepr --configs cifar-10_fedper_hca_2epr.yml cifar-10_fedper_hca_3epr.yml cifar-10_fedper_hca_5epr.yml
    ```
- Visualize the results:
    ```
    python src/plot.py --dataset cifar10 --folder cifar-10_vepr --cross_configuration_plots_and_tables --vepr --vepr_inputs CIFAR-10_FedPer_HCA_2epr CIFAR-10_FedPer_HCA_3epr CIFAR-10_FedPer_HCA_5epr
    ```
### CelebA
The CelebA dataset is used to conduct multi-label image classification tasks, as well as face landmark point detection tasks.
The resulting two task groups, model task heterogenity. On this dataset, 3 different experiments, that in total required the training of 10 federations (resulting in 60 models), were conducted. To save on computing time, the experiments use only a subset of the CelebA dataset. For details about the specific setups, please see the respective config files.

#### CelebA-10 with Different Aggregation Schemes
This experiment compares the proposed framework with pure local training, backbone averaging within the taskgroup, as well as with backbone averaging within and across task groups. (Approx 4h on GeForce 3080)
- Run the federation:
    ```
    python src/run.py --configs_folder configs/celeba --configs celeba_none_none.yml celeba_fedper_none.yml celeba_fedper_fedper.yml celeba_fedper_hca.yml
    ```
- Visualize the results:
    ```
    python src/plot.py --dataset celeba --folder celeba --cross_configuration_plots_and_tables --vag --vag_inputs CelebA_None_None CelebA_FedPer_None CelebA_FedPer_FedPer CelebA_FedPer_HCA
    ```

#### CelebA With Varying Backbone Layers
This experiment runs various configurations of the proposed framework, with varying the backbone/head split. (Approx 3h on GeForce 3080)
- Run the federation:
    ```
    python src/run.py --configs_folder configs/celeba_vbl --configs celeba_fedper_hca_bl-0.yml celeba_fedper_hca_bl-1.yml celeba_fedper_hca_bl-2.yml
    ```
- Visualize the results:
    ```
    python src/plot.py --dataset celeba --folder celeba_vbl --cross_configuration_plots_and_tables --vbl --vbl_inputs CelebA_FedPer_HCA_BL-0 CelebA_FedPer_HCA_BL-1 CelebA_FedPer_HCA_BL-2
    ```

#### CelebA With Varying Epochs per Round
This experiment runs various configurations of the proposed framework, with varying amounts of epochs between aggregation rounds. (Approx 3h on GeForce 3080)
- Run the federation:
    ```
    python src/run.py --configs_folder configs/celeba_vepr --configs celeba_fedper_hca_2epr.yml celeba_fedper_hca_3epr.yml celeba_fedper_hca_5epr.yml
    ```
- Visualize the results:
    ```
    python src/plot.py --dataset celeba --folder celeba_vepr --cross_configuration_plots_and_tables --vepr --vepr_inputs CelebA_FedPer_HCA_2epr CelebA_FedPer_HCA_3epr CelebA_FedPer_HCA_5epr
    ```