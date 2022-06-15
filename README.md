# Introduction

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
pre-commit install -c pre-commit-config.yaml
pre-commit install -c pre-merge-config.yaml -t pre-merge-commit

```
If you are using a proxy
```
pip install --proxy http://ccsmtl.proxy.mtl.rtss.qc.ca -r requirement.txt
```
## Proxy help
```
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
## Does not go there

[create an account](https://wandb.ai/site) on weight and biases than ask to join the project
#### In settings
go to settings->network->proxy

#### In the environment

You also need to set the following environment variables :
```
HTTP_PROXY={proxy_url:port}
HTTPS_PROXY={proxy_url:port
```

#### For apt

apt requires its own configuration file
```
sudo touch /etc/apt/apt.conf.d/proxy.conf
```

```
sudo nano /etc/apt/apt.conf.d/proxy.conf
```

and add to the file :

```
Acquire::http::Proxy "http://user:password@proxy.server:port/";
Acquire::https::Proxy "http://user:password@proxy.server:port/";
```
#### for pip
pip install takes the flag --proxy

#### for wget, curl, etc
those require different proxy config files. Remember Google is your friend
## Create a virtual work environnement
install python 3.9
```
python3.9 -m venv venv
```
```
source venv/bin/activate
```
## Install the dependencies
pip install --proxy {proxy} -r requirements.txt

## Setting up git hooks

This will execute test before commiting to main
``
git config core.hooksPath .githooks
``
## Download the data
A data manager has been provided to both download the data and extract it .
These require wget to be configure properly.
run
````
data_api/data_downloader.py
````


````
data_api/data_organizer.py
````
# Build and Test

## Build
TODO: Describe and show how to build your code and run the tests.

sudo pip install --proxy http://ccsmtl.proxy.mtl.rtss.qc.ca:8080 -r requirements.txt
python train.py --model densenet201

## Test
The test can be found in test.py.
Run them by executing

pytest -v test.py
# Contribute
TODO: Explain how other users and developers can contribute to make your code better.

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
