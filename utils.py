import subprocess
import sys
from typing import Optional

VAST_NUM = 5
VAST_PORT = 14902
SSH_DIRECTORY = "sparse_coding"
dest_addr = f"root@ssh{VAST_NUM}.vast.ai"

def sync():
    """ Sync the local directory with the remote host."""
    command = f"rsync -rv --filter \":- .gitignore\" --exclude \".git\" -e \"ssh -p {VAST_PORT}\" . {dest_addr}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)

    # also copy across secrets.json
    command = f"scp -P {VAST_PORT} secrets.json {dest_addr}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)

def copy_models():
    """ Copy the models from local directory to the remote host."""
    command = f"scp -P {VAST_PORT} -r models {dest_addr}:{SSH_DIRECTORY}/models"
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    if sys.argv[1] == "sync":
        sync()
    elif sys.argv[1] == "models":
        copy_models()


class dotdict(dict):
    # if no dict given, create empty dict
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
    