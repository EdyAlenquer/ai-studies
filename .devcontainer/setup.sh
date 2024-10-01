#!/bin/bash

# install dev requirements
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --timeout 100 --no-cache-dir -r ./.devcontainer/dev-requirements.txt

# Install Ollama and pull llama3.1:8b
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3.1:8b

# cd /tmp

# install docker to interact with the host if you plan to use remote docker inside the devcontainer
# for example to train a model on the host but in a container, too
# curl -fsSL https://get.docker.com -o get-docker.sh
# sh get-docker.sh

# install aws cli to interact with the aws cli from the devcontainer
# don't forget to configure the aws cli with your credentials inside devcontainer.json
# apt-get install zip -y
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# ./aws/install