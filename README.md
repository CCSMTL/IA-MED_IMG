_# Introduction

TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. And give the main sources

# Getting Started
## Install on Windows

* [Visual Studio Code](https://code.visualstudio.com/)

* [python3.9](https://www.python.org/downloads/release/python-3913/) (optional)

* python extension for VS code

* [Github Desktop](https://www.python.org/downloads/release/python-3913/)

* [git](https://git-scm.com/downloads)


## Set up Wsl
Go the search bar in the bottom left of youre screen and type : Turn Windows features on or off. \
Activate : Windows subsystem for Linux \
Activate : Plateforme de l'hyperviseur Windows\
Got to Microsoft Store and download Wsl Debian and open it to finish installation\
Still from Microsoft store download ubuntu 20.04.4 LTS\
Launch ubuntu \
Set username and password \
install [python](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu) on unix \
install pip
```
    sudo apt update
    sudo apt install python3-pip
```
## On Windows
open github desktop and clone the project's repository using Url :
```
     http://csrsvr00116:8080/DefaultCollection/IA-MED_IMG/_git/IA-MED_IMG
```
Use the short key Ctrl + shift + A to directly open the project in VS Code \
In VS Code open a new terminal and on the top right of this terminal beside powershell select Debian (WSL). From there you will download packages in ubuntu and run everything in ubuntu WSL.\
Install requirements
```
pip install -r requirements.txt
```

for development, also run
```
pip install -r requirements_dev.txt
python -m pip install -e .

```
If you are using a proxy
```
pip install --proxy http://ccsmtl.proxy.mtl.rtss.qc.ca -r requirement.txt
```
## Proxy help
```
sudo apt-get  -o Acquire::http::proxy=http://ccsmtl.proxy.mtl.rtss.qc.ca:8080 install <package>


export HTTP_PROXY="http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
export HTTPS_PROXY="http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"

unset HTTP_PROXY
unset HTTPS_PROXY

git config --global http.proxy http://ccsmtl.proxy.mtl.rtss.qc.ca:8080
git config --global https.proxy http://ccsmtl.proxy.mtl.rtss.qc.ca:8080
git config --add remote.origin.proxy ""
git config --global --unset http.proxy
git config --global --unset https.proxy

pip config set global.proxy http://ccsmtl.proxy.mtl.rtss.qc.ca:8080
```
## Experiment Tracking - W&B

[create an account](https://wandb.ai/site) on weight and biases than ask to join the [organization](https://wandb.ai/ccsmtl)


## Setting the environment

### In settings
go to settings->network->proxy

### In the environment

You also need to set the following environment variables :
```
HTTP_PROXY={proxy_url:port}
HTTPS_PROXY={proxy_url:port
```

### For apt

apt requires its own configuration file
```
sudo touch /etc/apt/apt.conf.d/proxy.conf
```

```
sudo nano /etc/apt/apt.conf.d/proxy.conf
```

and add to the file :

```
Acquire::http::Proxy "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080";
Acquire::https::Proxy "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080";
```

To use apt without proxy after without changing the config file :
```
sudo apt-get -o Acquire::http::proxy=false <update/install> 
```
### for pip
pip install takes the flag --proxy

### for wget, curl, etc
those require different proxy config files. Remember Google is your friend
### Create a virtual work environnement
install python 3.9
```
python3.9 -m venv venv
```
```
source venv/bin/activate
```
### Install the dependencies

```
pip install --proxy {proxy} -r radia/requirements.txt
pip install --proxy {proxy} -r requirements_dev.txt
pip install -e .

```

``
### Download the data (just Chexpert mini for debug purpose)
A data manager has been provided to both download the data and extract it .
These require wget to be configure properly.
run
````
data_api/data_downloader.py
````


````
data_api/data_organizer.py
````


## Test
The test can be found in the folder tests.
Run them by executing

```
pytest -v ./tests
```

## Commits

To be able to commit on master, you will need to
do pull requests. Those pull requests will then need to be reviewed by 2 users (can include yourself)
Here is how to proceed 

1) Create a new branch 
git checkout -b new_branch
2) commit some changes
git commit ... -m "commit message"
3) push those to the origin
git push -u origin
4) Create pull request
In azure Devops, go to the repos and select pull request. Your changes should be available there
in order to finalize the pull request. Please take notes that all the tests run by pytest need to work before 
the pull request is allowed to move forward.



# File structure
````
..
├── .gitignore
├── .pytest-deps
├── radia
│   ├── Experiment.py
│   ├── Metrics.py
│   ├── Parser.py
│   ├── Transforms.py
│   ├── __init__.py
│   ├── custom_utils.py
│   ├── data analysis
│   │   └── clustering.py
│   ├── data_API
│   │   ├── create_test_set.py
│   │   ├── data_analysis.py
│   │   ├── data_downloader_chestXray.py
│   │   ├── data_organizer.py
│   │   ├── download.sh
│   │   └── links.txt
│   ├── data_exploration.ipynb
│   ├── data_visualization.py
│   ├── dataloaders
│   │   ├── CXRLoader.py
│   │   └── MongoDB.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── detect.py
│   │   ├── inference.ipynb
│   │   └── utils.py
│   ├── models
│   │   ├── CNN.py
│   │   ├── Ensemble.py
│   │   ├── Unet.py
│   │   ├── pylon.py
│   │   └── teacher_learning.py
│   ├── reproducing_libauc.ipynb
│   ├── requirements.txt
│   ├── results_visualization.py
│   ├── run.sh
│   ├── sanity_check.ipynb
│   └── training
│       ├── Loss.py
│       ├── __init__.py
│       ├── multi_gpu_train.py
│       ├── train.py
│       └── training.py
├── Pipfile
├── README.md
├── azure-pipelines.yml
├── data
│   ├── chexnet_results.csv
│   └── data_table.csv
├── plots
│   ├── Confusion Matrix.png
│   ├── Most_Frequented_Classes111.png
│   ├── chords_chexpert.png
│   ├── chords_mongodb.png
│   ├── histogram_chexpert.png
│   └── histogram_mongodb.png
├── requirements_dev.txt
├── setup.py
├── tests
│   ├── __init__.py
│   ├── data_test
│   │   ├── CheXpert-v1.0-small
│   │   │   └── valid
│   │   │       ├── patient64541
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       ├── patient64542
│   │   │       │   └── study1
│   │   │       │       ├── view1_frontal.jpg
│   │   │       │       └── view2_lateral.jpg
│   │   │       ├── patient64543
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       ├── patient64544
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       ├── patient64545
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       ├── patient64546
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       ├── patient64547
│   │   │       │   └── study1
│   │   │       │       ├── view1_frontal.jpg
│   │   │       │       ├── view2_frontal.jpg
│   │   │       │       └── view3_lateral.jpg
│   │   │       ├── patient64548
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       ├── patient64549
│   │   │       │   └── study1
│   │   │       │       └── view1_frontal.jpg
│   │   │       └── patient64550
│   │   │           └── study1
│   │   │               └── view1_frontal.jpg
│   │   ├── data.yaml
│   │   ├── images
│   │   │   ├── 00027725_000.png
│   │   │   ├── 00027725_001.png
│   │   │   ├── 00027725_002.png
│   │   │   ├── 00027725_003.png
│   │   │   ├── 00027725_004.png
│   │   │   ├── 00027725_005.png
│   │   │   ├── 00027725_006.png
│   │   │   ├── 00027725_007.png
│   │   │   ├── 00027725_008.png
│   │   │   └── 00027725_009.png
│   │   ├── labels
│   │   │   ├── 00027725_000.txt
│   │   │   ├── 00027725_001.txt
│   │   │   ├── 00027725_002.txt
│   │   │   ├── 00027725_003.txt
│   │   │   ├── 00027725_004.txt
│   │   │   ├── 00027725_005.txt
│   │   │   ├── 00027725_006.txt
│   │   │   ├── 00027725_007.txt
│   │   │   ├── 00027725_008.txt
│   │   │   └── 00027725_009.txt
│   │   ├── sampler_weights.txt
│   │   ├── train.csv
│   │   └── valid.csv
│   ├── run_black.sh
│   ├── run_pylint.sh
│   ├── test_.py
│   ├── test_Experiment.py
│   ├── test_cnn.py
│   ├── test_cxrloader.py
│   └── test_train.py
└── visualization.py



````

## Training Scripts

### Initial Setup

Make sure the environment variable img_dir is defined and points to the folder containing /data/...
where the images are stored

Make sure you have logged into your W&B account.

You can use the command wandb online/wandb offline to toggle online/offline mode.

### train.py

Used to launch training on a single GPU/CPU
Instruction on specific arguments can be obtained by running
``
python radia/training/train.py --help
``

A simplified context manager has been created to run experience. The class Experiment.py allow to easily compile run by specifying the desired optimizer and 
loss "à la Keras". The class is defined in Radia/Experiment.py

Please note a few element are still hard-coded in the class. These are 

- The names and hierarchy of the classes : see Radia/__init__.py to customize these
- the connection to the database : see Radia/dataloaders/CXRLoader.py to customize the IP address of the Mongo Database 
- The positive weights associated for each classes : They are calculated in the dataloader : see Radia/dataloaders/CXRLoader.py
- The sample weight of each element in the dataloader : They are calculated in the dataloader also : see Radia/dataloaders/CXRLoader.py
- The datasets used : see Radia/training/train.py to change which dataset is used for either training or pretraining
## multi_gpu_train.py

Used to launch training on multiple GPUs. Still unstable . Use with caution as it is not yet functional. The synchronisation of gradient seem broken




## Tips and tricks

### Tmux

tmux is a terminal multiplexer. It lets you switch easily between several programs in one terminal, detach them (they keep running in the background) and reattach them to a different terminal.
It is extremely useful to run a program for a long time without worrying the 
SSH connection is cut.


