import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from clsr.runner import SimpleRunner

random.seed(45867496)
np.random.seed(45867496)
torch.manual_seed(45867496)
torch.cuda.manual_seed(45867496)


@hydra.main(config_path='../configs', config_name='autoencoder')
def main(cfg: DictConfig):
    runner = SimpleRunner(cfg, mode=1)  # test mode
    runner.test()


if __name__ == '__main__':
    main()
