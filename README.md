This repository is a PyTorch DistributedDataParallel (DDP) re-implementation of the CVPR 2020 paper 
[View-GCN](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_View-GCN_View-Based_Graph_Convolutional_Network_for_3D_Shape_Analysis_CVPR_2020_paper.html). 

First, the re-implementation aims to accelerate the process of training and inference by the PyTorch DDP mechanism since
the [original implementation](https://github.com/weixmath/view-GCN) by the author is only for single-GPU learning
and the procedure is much slower, especially when reproducing the retrieval results on SHREC17 benchmark. 

Second, the retrieval code was absent in the [original repository](https://github.com/weixmath/view-GCN) and 
the author only released the classification code on ModelNet40. Our re-implementation adds the `retrieval experiment` and corresponding instructions. 

Third, we also add the `classification` code on the `RGBD` dataset, modify the model and arguments definition to adapt multi-dataset training, 
rewrite the READEME and optimzie code style, etc.

In summary, this repository has the following new features compared to the original one

## New Features
1. add DDP acceleration for the model training and inference
2. add the classification code on the RGBD dataset
3. add the retrieval code on the SHREC17 dataset
4. modify the model definition and add arguments to adapt multi-dataset training and inference 
5. upgrade the PyTorch and torchvision to recent version 1.12.0 and 0.13.0, respectively, and give several fixes
6. re-organize README and optimize the code style

## Preparation
### Package Setup
* Ubuntu 18.04
* Python 3.7.15
* PyTorch 1.12.0
* CUDA 11.6
* torchvision 0.13.0
* timm 0.6.11
* einops 0.6.0
* wandb 0.12.11
* pueue & pueued 2.0.4

```shell
  conda create -n viewgcn python=3.7.15
  codna activate viewgcn

  pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
  pip install -r requirements.txt
```

`pueue` is a shell command management software, we use it for scheduling the model training & inference tasks, please refer to the [official page](https://github.com/Nukesor/pueue) for installation and basic usage. 
We recommend this tool because under its help you can run the experiments at scale thus save your time. 

### W&B Server Setup
We track the model training and fine-tuning with W&B tools. The official W&B tools may be slow and unstable since 
they are on remote servers, we install the local version by running the following command. 

```shell
  docker run --rm -d -v wandb:/vol -p 28282:8080 --name wandb-local wandb/local:0.9.41
```

If you do not have Docker installed on your computer before, referring to the [official document](https://docs.docker.com/engine/install/ubuntu/) to finish Docker installation on Ubuntu.

### Datasets
1. Download the following datasets and extract them to the desired location on your computer. 
    1. [ModelNet10](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet10v2png_ori2.tar)
    2. [ModelNet40](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar)
    3. [RGBD](https://rgbd-dataset.cs.washington.edu/dataset/)
    4. [SHREC17](https://shapenet.cs.stanford.edu/shrec17/)
        - follow [this project](https://github.com/kanezaki/SHREC2017_track3) to render the meshes to get the multiple views

2. The directories of the above datasets should be organized as follow 
    ```
    |- View-GCN-DDP
    |---- data
    |-------- ModelNet10
    |-------- ModelNet40
    |-------- RGBD
    |-------- SHREC17
    ```
    The `data` directory is at the same level with `models`, `scripts`, etc.

## Usage
### 3D Shape Recognition
1. To train and evaluate on ModelNet10, run
    ```shell
    ./scripts/MN10-V20-L4H8D512-MR2-Alex-1.sh
    ```

2. To train and evaluate on ModelNet40, run
    ```shell
    ./scripts/MN40-V20-L4H8D512-MR2-Alex-1.sh
    ```

3. To train and evaluate on RGBD, run
    ```shell
    ./scripts/RGBD-V12-L4H8D512-MR2-Alex-1.sh
    ```

### 3D Shape Retrieval
1. To train and evaluate the classification performance on SHREC17, run
    ```shell
    ./scripts/RET/viewgcn_shrec17/SH17-V20-ViewGCN-RN18-1.sh
    ```

2. Change the work directory and make a new directory that are used to save the retrieval results
    ```
    cd retrieval
    mkdir -p evaluator/viewgcn
    ```

3. Retrieve shapes that have same class as the query to generate the rank list
    ```
    python shrec17.py val.csv resnet18 0 SH17-V20-ViewGCN-RN18-1 24 viewgcn
    ```

4. Evaluate the retrieval performance
    ```
    node --max-old-space-size=8192 evaluate.js viewgcn/
    ```

5. Replace `val.csv` with `test.csv` and re-run steps 3-6 to get the results of the test split