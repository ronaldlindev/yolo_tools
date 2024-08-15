import os
import yaml

## WIP
class yololoader:

    def __init__(self, yaml_path: str):
        with open(yaml_path) as f:
            self.data = yaml.load(f)
        self.img_idx = 0
        self.dir_idx = 0
        # find all image dirs
                



    def __iter__(self):
        return self

    def __next__(self):
        pass
        
