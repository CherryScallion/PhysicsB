## utils/paths.py
##ps: I dont think this file is nessesary, and it makes any change in modules difficult.
import os
from pathlib import Path

def get_project_root() -> Path:
    """
    aquire the project root directory
    """
    # assume utils/paths.py is in [ROOT]/utils/paths.py
    return Path(__file__).resolve().parent.parent

def resolve_path(relative_path: str) -> Path:
    """revert relative path to absolute path based on project root"""
    return get_project_root() / relative_path

def get_config_path() -> Path:
    return get_project_root() / "configs" / "train_config.yaml"

def get_checkpoint_dir() -> Path:
    return get_project_root() / "checkpoints"

def get_template_dir() -> Path:
    return get_project_root() / "data" / "templates"

if __name__ == "__main__":
    print(f"Project Root Detected: {get_project_root()}")