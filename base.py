import os
from abc import ABC, abstractmethod

from utils import  make_dir


class BaseModel(ABC):
    def __init__(self, device, output_dir, opts):
        self.output_dir = output_dir
        make_dir(self.output_dir)

        self.model_dir = os.path.join(self.output_dir, 'model')
        self.image_dir = os.path.join(self.output_dir, 'image')
        make_dir(self.model_dir)
        make_dir(self.image_dir)

        self.device = device
        self.opts = opts

    @abstractmethod
    def evaluate(self, model_path, test_loader):
        pass

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad