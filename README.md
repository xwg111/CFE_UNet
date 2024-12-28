# CFE-UNet
Our network structure is shown in the figure
![CFE-UNet](docs/DAC-Net.png)


## Main Environments

- torch~=2.1.2+cu121
- numpy~=1.23.2
- opencv-python~=4.6.0.66
- scikit-learn~=1.4.1.post1
- nets~=0.0.3.1
- thop~=0.1.1-2209072238
- matplotlib~=3.8.0
- tqdm~=4.61.2
- utils~=1.0.2
- tensorboardX~=2.4
- torchvision~=0.16.2+cu121
- scipy~=1.12.0
- chardet~=5.2.0
- timm~=0.9.16
- einops~=0.3.0
- loguru~=0.7.2
- scikit-image~=0.17.2


## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```


#### Then prepare the datasets in the following format for easy use of the code:

```
├── datasets
    ├── Glas
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── BUSI
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```
