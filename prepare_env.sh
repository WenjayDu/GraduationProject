#!/bin/bash

# global variables
USER_NAME=`whoami`

# update and install dependencies
sudo apt update && \
sudo apt upgrade -y && \
sudo apt install python3-pip jupyter-core -y && \
sudo ln -s /usr/bin/python3 /usr/bin/python && \
sudo ln -s /usr/bin/pip3 /usr/bin/pip && \
sudo pip3 install -r requirements.txt

# config jupyter and enable remote access
jupyter notebook --generate-config && \
jupyter notebook password && \
mkdir ~/jupyter_root && \
(echo '0a'; echo "
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.notebook_dir = u'/home/$USER_NAME/jupyter_root'
c.NotebookApp.open_browser = False
"; echo '.'; echo 'wq') | ed -s ~/.jupyter/jupyter_notebook_config.py

# install cuda
# take installing CUDA v10.0 on Ubuntu_18.04_X86_64 as an example
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64 -O cuda.deb && \
sudo dpkg -i cuda.deb && \
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub && \
sudo apt update && \
sudo apt install cuda -y && \
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc && \
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc && \
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc && \
source ~/.bashrc && \
nvidia-smi

# install cuDNN
# you can get cudnn v10.0 in the url as follows:
# https://developer.nvidia.com/rdp/cudnn-archive#a-collapse750-10
# can use `gcloud scp` to upload cudnn to instance, you can execute the command similar with the following:
# gcloud compute scp cudnn-10.0-linux-x64-v7.5.0.56.tgz username@instance_name:/home/username

while [[ ! -f "~/cudnn.tgz" ]];
do
    echo "error: ~/cudnn.tgz does not exist in `pwd`, waiting for your uploading..."
    sleep 30
done

echo "Start installing cuDNN..."
tar -zxvf cudnn.tgz && \
sudo mv cuda/lib64/* /usr/local/cuda/lib64/ && \
sudo mv cuda/include/cudnn.h /usr/local/cuda/include/ && \
rm -rf ~/cuda

# install tensorflow-gpu
sudo pip install --upgrade tensorflow-gpu