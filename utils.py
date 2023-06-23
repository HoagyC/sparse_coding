import json
import os
import subprocess
import sys
from typing import Optional
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError

VAST_NUM = 4
VAST_PORT = 16356
SSH_DIRECTORY = "sparse_coding"
dest_addr = f"root@ssh{VAST_NUM}.vast.ai"
SSH_PYTHON = "/opt/conda/bin/python"

BUCKET_NAME = "sparse-coding"

ACCESS_KEY_NAME_DICT = {
    "AKIAV3IKT32M2ZA3WRLQ": "hoagy",
    "AKIATUSYDLZAEZ7T5GLX": "aidan",
    "AKIATEQID7TUM5FUW4R5": "logan",
}

def sync():
    """Sync the local directory with the remote host."""
    command = f'rsync -rv --filter ":- .gitignore" --exclude ".git" -e "ssh -p {VAST_PORT}" . {dest_addr}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)


def copy_models():
    """Copy the models from local directory to the remote host."""
    command = f"scp -P {VAST_PORT} -r models {dest_addr}:{SSH_DIRECTORY}/models"
    subprocess.call(command, shell=True)
    # also copying across a few other files
    command = f"scp -P {VAST_PORT} -r outputs/thinrun/autoencoders_cpu.pkl {dest_addr}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)

def copy_secrets():
    """Copy the secrets.json file from local directory to the remote host."""
    command = f"scp -P {VAST_PORT} secrets.json {dest_addr}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)


def copy_recent():
    """Get the most recent outputs folder in the remote host and copy across to same place in local directory."""
    # get the most recent folders
    command = f'ssh -p {VAST_PORT} {dest_addr} "ls -td {SSH_DIRECTORY}/outputs/* | head -1"'
    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8").strip()
    # copy across
    command = f"scp -P {VAST_PORT} -r {dest_addr}:{output} outputs"
    subprocess.call(command, shell=True)


def setup():
    """Sync, copy models, create venv and install requirements."""
    sync()
    copy_models()
    copy_secrets()
    command = f'ssh -p {VAST_PORT} {dest_addr} "cd {SSH_DIRECTORY} && {SSH_PYTHON} -m venv .env && source .env/bin/activate && pip install -r requirements.txt" && apt install vim'
    # command = f"ssh -p {VAST_PORT} {dest_addr} \"cd {SSH_DIRECTORY} && echo $PATH\""
    subprocess.call(command, shell=True)
    # clone neuron explainer, until i can load it from pip
    command = f'ssh -p {VAST_PORT} {dest_addr} "cd sparse_coding && git clone https://github.com/openai/automated-interpretability && mv automated-interpretability/neuron-explainer/neuron_explainer/ neuron_explainer"'
    subprocess.call(command, shell=True)

class dotdict(dict):
    """Dictionary that can be accessed with dot notation."""

    def __init__(self, d: Optional[dict] = None):
        if d is None:
            d = {}
        super().__init__(d)

    def __dict__(self):
        return self

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

def make_tensor_name(cfg):
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        tensor_name = f"blocks.{cfg.layer}.mlp.hook_post"
        if cfg.model_name == "gpt2":
            cfg.mlp_width = 3072
        elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
            cfg.mlp_width = 2048
    elif cfg.model_name == "nanoGPT":
        tensor_name = f"transformer.h.{cfg.layer}.mlp.c_fc"
        cfg.mlp_width = 128
    else:
        raise NotImplementedError(f"Model {cfg.model_name} not supported")

    return tensor_name

def upload_to_aws(local_file_name) -> bool:
    """"
    Upload a file to an S3 bucket
    :param local_file_name: File to upload
    :param s3_file_name: S3 object name. If not specified then local_file_name is used
    """
    secrets = json.load(open("secrets.json"))

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )

    if secrets["access_key"] in ACCESS_KEY_NAME_DICT:
        name = ACCESS_KEY_NAME_DICT[secrets["access_key"]]
    else:
        name = "unknown"

    s3_file_name = name + "-" + local_file_name

    local_file_path = Path(local_file_name)
    try:
        if local_file_path.is_dir():
            _upload_directory(local_file_name, s3)
        else:
            s3.upload_file(str(local_file_name), BUCKET_NAME, str(s3_file_name))
        print(f"Upload Successful of {local_file_name}")
        return True
    except FileNotFoundError:
        print(f"File {local_file_name} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))
    

if __name__ == "__main__":
    if sys.argv[1] == "sync":
        sync()
    elif sys.argv[1] == "models":
        copy_models()
    elif sys.argv[1] == "recent":
        copy_recent()
    elif sys.argv[1] == "setup":
        setup()
    elif sys.argv[1] == "secrets":
        copy_secrets()
