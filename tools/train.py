import random

import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from clsr.runner import SimpleRunner

random.seed(45867496)
np.random.seed(45867496)
torch.manual_seed(45867496)
torch.cuda.manual_seed(45867496)


@hydra.main(config_path='../configs', config_name='autoencoder')
def main(cfg: DictConfig):
    runner = SimpleRunner(cfg, mode=0) # train_val mode
    runner.train()


if __name__ == '__main__':
    main()
