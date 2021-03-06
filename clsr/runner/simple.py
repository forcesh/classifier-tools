import copy
import math
import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from clsr.loss.builder import build_loss
from clsr.model.builder import build_model
from clsr.optimizer.builder import build_optimizer
from clsr.settings import PROJECT_ROOT


class SimpleRunner:
    def __init__(self, cfg: DictConfig, mode: int = 0):
        self.cfg = cfg
        self.mode = mode  # 0 - train_val, 1 - test
        self.device = cfg.runner.device

        self._init_model()
        self._init_loaders()
        self.criterion = build_loss(cfg.loss)
        self.optimizer = None
        if self.mode == 0:
            self.optimizer = build_optimizer(cfg.optimizer.name,
                                             filter(lambda p: p.requires_grad,
                                                    self.model.parameters()),
                                             lr=cfg.optimizer.lr)

    def _init_model(self):
        kwargs = {}
        if hasattr(self.cfg.model, 'input_dim'):
            kwargs['input_dim'] = self.cfg.model.input_dim
        if hasattr(self.cfg.model, 'hidden_dim'):
            kwargs['hidden_dim'] = self.cfg.model.hidden_dim
        if hasattr(self.cfg.model, 'num_classes'):
            kwargs['num_classes'] = self.cfg.model.num_classes
        if hasattr(self.cfg.model, 'encoder_weights'):
            if self.cfg.model.encoder_weights == 'None' or self.mode != 0:
                pass
            else:
                encoder_weights = self.cfg.model.encoder_weights
                kwargs['encoder_weights'] = os.path.join(
                    PROJECT_ROOT, encoder_weights)
        self.model = build_model(self.cfg.model.name, **kwargs).to(self.device)

        if self.mode == 1:
            weights_path = os.path.join(PROJECT_ROOT,
                                        self.cfg.inference.weights)
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint)
            print(weights_path, 'weights loaded')
        else:
            if hasattr(self.cfg.model, 'weights'):
                weights_path = os.path.join(PROJECT_ROOT,
                                            self.cfg.inference.weights)
                if os.path.exist(weights_path):
                    checkpoint = torch.load(weights_path)
                    self.model.load_state_dict(checkpoint)
                    print(weights_path, 'weights loaded')

    def _init_loaders(self):
        if self.cfg.data.dataset == 'mnist':
            dataset_dir = os.path.join(PROJECT_ROOT,
                                       self.cfg.runner.dataset_dir)

            # ToDo build_transforms by cfg
            main_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])

            trainset = torchvision.datasets.MNIST(root=dataset_dir,
                                                  train=True,
                                                  download=False,
                                                  transform=train_transform)
            testset = torchvision.datasets.MNIST(root=dataset_dir,
                                                 train=False,
                                                 download=False,
                                                 transform=main_transform)

            trainloader = DataLoader(trainset,
                                     batch_size=self.cfg.runner.batch_size,
                                     shuffle=True)
            valloader = DataLoader(testset,
                                   batch_size=self.cfg.inference.batch_size,
                                   shuffle=False)

            self.loaders = {
                'train': trainloader,
                'valid': valloader,
                'test': valloader,  # ToDo make test set
            }
        else:
            raise NotImplementedError

    def train(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        if self.cfg.model.name == 'autoencoder':
            best_metric = -math.inf
        else:
            best_metric = 0.0
        os.makedirs(os.path.join(PROJECT_ROOT, self.cfg.runner.logdir,
                                 'checkpoints'),
                    exist_ok=True)
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
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        if self.cfg.model.name == 'autoencoder':
                            loss = self.criterion(outputs, inputs)
                            metric = -loss.item() * inputs.size(0)
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
                epoch_metric = running_metric / len(
                    self.loaders[phase].dataset)

                print('{} Loss: {:.4f} Metric(MSE or Acc): {:.4f}'.format(
                    phase, epoch_loss, epoch_metric))

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

    def test_autoencoder(self):
        running_loss = 0.0
        for i, batch in enumerate(tqdm(self.loaders['test'])):
            inputs, labels = batch
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                running_loss += loss.item() * inputs.size(0)
        mse = running_loss / len(self.loaders['test'].dataset)
        return -mse

    def test_classifier(self):
        gt_labels = []
        pred_labels = []

        for i, batch in enumerate(tqdm(self.loaders['test'])):
            inputs, labels = batch
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                gt_labels.extend(labels.cpu().numpy().tolist())
                pred_labels.extend(predicted.cpu().numpy().tolist())

        print(classification_report(gt_labels, pred_labels))
        return f1_score(gt_labels, pred_labels, average='micro')

    def test(self):
        self.model.eval()
        if self.cfg.model.name == 'autoencoder':
            metric = self.test_autoencoder()
            print('mse: {:.4f}'.format(-metric))
        else:
            metric = self.test_classifier()
            print('f1_score: {:.4f}'.format(metric))
        return metric
