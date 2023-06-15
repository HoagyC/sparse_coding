import subprocess
import sys
from typing import Optional

VAST_NUM = 4
VAST_PORT = 13158
SSH_DIRECTORY = "sparse_coding"
dest_addr = f"root@ssh{VAST_NUM}.vast.ai"
SSH_PYTHON = "/opt/conda/bin/python"


def sync():
    """Sync the local directory with the remote host."""
    command = f'rsync -rv --filter ":- .gitignore" --exclude ".git" -e "ssh -p {VAST_PORT}" . {dest_addr}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)


def copy_models():
    """Copy the models from local directory to the remote host."""
    command = f"scp -P {VAST_PORT} -r models {dest_addr}:{SSH_DIRECTORY}/models"
    subprocess.call(command, shell=True)


def copy_secrets():
    """Copy the secrets.json file from local directory to the remote host."""
    command = f"scp -P {VAST_PORT} secrets.json {dest_addr}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)


def copy_recent():
    """Get the most recent outputs folder in the remote host and copy across to same place in local directory."""
    # get the most recent folder
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
    command = f'ssh -p {VAST_PORT} {dest_addr} "git clone https://github.com/openai/automated-interpretability && mv automated-interpretability/neuron-explainer/neuron_explainer/ neuron_explainer"'
    subprocess.call(command, shell=True)

class dotdict(dict):
    """Dictionary that can be accessed with dot notation."""

    def __init__(self, d: Optional[dict] = None):
        if d is None:
            d = {}
        super().__init__(d)
        self.__dict__ = self

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
