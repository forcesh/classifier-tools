import os
import copy
import time

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from clsr.optimizer.builder import build_optimizer
from clsr.loss.builder import build_loss
from clsr.model.builder import build_model
from clsr.settings import PROJECT_ROOT


class SimpleRunner:
    def __init__(self,
                 cfg: DictConfig,
                 mode: int = 0):
        self.cfg = cfg
        self.mode = mode # 0 - train_val, 1 - test
        self.device = cfg.runner.device

        self._init_model()
        self._init_loaders()
        self.criterion = build_loss(cfg.loss)
        self.optimizer = None
        if self.mode == 0:
            self.optimizer = build_optimizer(cfg.optimizer.name,
                                             self.model.parameters(),
                                             lr=cfg.optimizer.lr)

    def _init_model(self):
        kwargs = {}
        if hasattr(self.cfg.model, "input_dim"):
            kwargs["input_dim"] = self.cfg.model.input_dim
        if hasattr(self.cfg.model, "hidden_dim"):
            kwargs["hidden_dim"] = self.cfg.model.hidden_dim
        if hasattr(self.cfg.model, "num_classes"):
            kwargs["num_classes"] = self.cfg.model.num_classes
        if hasattr(self.cfg.model, "encoder_weights"):
            if self.cfg.model.encoder_weights == "None":
                encoder_weights = None
            else:
                encoder_weights = self.cfg.model.encoder_weights
            kwargs["encoder_weights"] = encoder_weights
        self.model = build_model(self.cfg.model.name, **kwargs).to(self.device)

    def _init_loaders(self):
        if self.cfg.data.dataset == "mnist":
            dataset_dir = os.path.join(PROJECT_ROOT, self.cfg.runner.dataset_dir)
            print (dataset_dir)

            # ToDo build_transforms by cfg
            main_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            trainset = torchvision.datasets.MNIST(root=dataset_dir, train=True,
                                                  download=False, transform=train_transform)
            testset = torchvision.datasets.MNIST(root=dataset_dir, train=False,
                                                 download=False, transform=main_transform)

            trainloader = DataLoader(trainset, batch_size=self.cfg.runner.batch_size, shuffle=True)
            valloader = DataLoader(testset, batch_size=self.cfg.inference.batch_size, shuffle=False)

            self.loaders = {
                'train': trainloader,
                'valid': valloader,
                'test': valloader, # ToDo make test set
            }
        else:
            raise NotImplementedError

    def train(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0
        os.makedirs(os.path.join(PROJECT_ROOT, self.cfg.runner.logdir, 'checkpoints'), exist_ok=True)
        num_epochs = self.cfg.runner.num_epochs

        for epoch in range(num_epochs):
            print('Epoch {}/{} Lr {}'.format(
                epoch, num_epochs - 1, self.optimizer.param_groups[0]['lr']))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_metric = 0.0

                for batch in tqdm(self.loaders[phase]):
                    inputs, labels = batch
                    inputs = inputs.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        if self.cfg.model.autoencoder:
                            loss = self.criterion(outputs, inputs)
                            metric = -loss.item()
                        else:
                            loss = self.criterion(outputs, labels)
                            metric = (predicted == labels).sum().item()

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_metric += metric

                epoch_loss = running_loss / len(self.loaders[phase].dataset)
                epoch_metric = running_metric / len(self.loaders[phase].dataset)

                print('{} Loss: {:.4f} Metric(MSE or Acc): {:.4f}'.format(phase, epoch_loss, epoch_metric))

                # deep copy the model
                if phase == 'valid':
                    if epoch_metric > best_metric:
                        best_metric = epoch_metric
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(PROJECT_ROOT, self.cfg.runner.logdir,
                                         'checkpoints', 'best_weights.pth'))
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(PROJECT_ROOT, self.cfg.runner.logdir,
                                     'checkpoints', 'last_weights.pth'))

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val metric: {:4f}'.format(best_metric))
            self.model.load_state_dict(best_model_wts)

        return best_metric

    def test(self):
        self.model.eval()

        gt_labels = []
        pred_labels = []
        running_loss = 0.0

        for batch in tqdm(self.loaders['test']):
            inputs, labels = batch
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()

            with torch.inference_mode():
                outputs = self.model(inputs)
                if self.cfg.model.autoencoder:
                    loss = self.criterion(outputs, inputs)
                    running_loss += loss.item() * inputs.size(0)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    gt_labels.append(gt_labels)
                    pred_labels.append(predicted)

        # ToDo
