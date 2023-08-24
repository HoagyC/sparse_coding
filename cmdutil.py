import json
import os
import subprocess
import sys
from typing import Optional, Union, List
from pathlib import Path

VAST_NUM = 4
# DEST_ADDR = f"root@ssh{VAST_NUM}.vast.ai"
DEST_ADDR = "mchorse@216.153.50.63"
SSH_PYTHON = "/opt/conda/bin/python"

PORT = 22

USER = "aidan"

SSH_DIRECTORY = f"sparse_coding_{USER}"
BUCKET_NAME = "sparse-coding"

ACCESS_KEY_NAME_DICT = {
    "AKIAV3IKT32M2ZA3WRLQ": "hoagy",
    "AKIATUSYDLZAEZ7T5GLX": "aidan",
    "AKIATEQID7TUM5FUW4R5": "logan",
}

def sync():
    """Sync the local directory with the remote host."""
    command = f'rsync -rv --filter ":- .gitignore" --exclude ".git" -e "ssh -p {PORT}" . {DEST_ADDR}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)

def datasets_sync():
    """Sync .csv files with the remote host."""
    command = f'rsync -am --include "*.csv" --include "*.test" --exclude "*" -e "ssh -p {PORT}" . {DEST_ADDR}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)

def autointerp_sync():
    """Sync the local directory with the remote host's auto interp results, excluding hdf files."""
    command = f'rsync -r --exclude "*.hdf" --exclude "*.pkl" -e ssh {DEST_ADDR}:{SSH_DIRECTORY}/auto_interp_results . '
    print(command)
    subprocess.call(command, shell=True)

def copy_models():
    """Copy the models from local directory to the remote host."""
    command = f"scp -P {PORT} -r models {DEST_ADDR}:{SSH_DIRECTORY}/models"
    subprocess.call(command, shell=True)
    # also copying across a few other files
    command = f"scp -P {PORT} -r outputs/thinrun/autoencoders_cpu.pkl {DEST_ADDR}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)

def copy_secrets():
    """Copy the secrets.json file from local directory to the remote host."""
    command = f"scp -P {PORT} secrets.json {DEST_ADDR}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)


def copy_recent():
    """Get the most recent outputs folder in the remote host and copy across to same place in local directory."""
    # get the most recent folders
    command = f'ssh -p {PORT} {DEST_ADDR} "ls -td {SSH_DIRECTORY}/outputs/* | head -1"'
    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8").strip()
    # copy across
    command = f"scp -P {PORT} -r {DEST_ADDR}:{output} outputs"
    subprocess.call(command, shell=True)

def copy_dotfiles():
    """Copy dotfiles into remote host and run install and deploy scripts"""
    df_dir = f"dotfiles_{USER}"
    command = f"scp -P {PORT} -r ~/git/dotfiles {DEST_ADDR}:{df_dir}"
    subprocess.call(command, shell=True)
    command = f"ssh -p {PORT} {DEST_ADDR} 'cd ~/{df_dir} && ./install.sh && ./deploy.sh'"
    subprocess.call(command, shell=True)

def setup():
    """Sync, copy models, create venv and install requirements."""
    sync()
    copy_models()
    copy_secrets()
    command = f'ssh -p {PORT} {DEST_ADDR} "cd {SSH_DIRECTORY} && sudo apt -y install python3.9 python3.9-venv && python3.9 -m venv .env --system-site-packages && source .env/bin/activate && pip install -r requirements.txt" && apt install vim'
    # command = f"ssh -p {VAST_PORT} {dest_addr} \"cd {SSH_DIRECTORY} && echo $PATH\""
    subprocess.call(command, shell=True)

import warnings

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
    elif sys.argv[1] == "interp_sync":
        autointerp_sync()
    elif sys.argv[1] == "dotfiles":
        copy_dotfiles()
    elif sys.argv[1] == "datasets":
        datasets_sync()
    else:
        raise NotImplementedError(f"Command {sys.argv[1]} not recognised")