# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

## Setup a Linux virtual machine
If you are working from a machine using Linux, skip this step. 
TODO \
start /windows feature/Activate : windows subsystem for Linux & Activate : Windows Hyperviser Platform\
Got ot microsoft store & download ubuntu 20.04.4 LTS \
Launch ubuntu \
set username and password

### set up a proxy in wsl

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