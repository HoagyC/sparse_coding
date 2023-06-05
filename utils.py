import subprocess
import sys
from typing import Optional

VAST_NUM = 4
VAST_PORT = 37398
SSH_DIRECTORY = "sparse_coding"
dest_addr = f"root@ssh{VAST_NUM}.vast.ai"

def sync():
    """ Sync the local directory with the remote host."""
    command = f"rsync -rv --filter \":- .gitignore\" --exclude \".git\" -e \"ssh -p {VAST_PORT}\" . {dest_addr}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    if sys.argv[1] == "sync":
        sync()


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
    