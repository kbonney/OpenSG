import yaml
import numpy as np


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        mesh_data = yaml.load(file, Loader=yaml.CLoader)
    return mesh_data

def write_yaml(data, yaml_file):
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, Loader=yaml.CLoader)
    return
