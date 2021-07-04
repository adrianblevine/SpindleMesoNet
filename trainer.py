""" adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils

from torchsummary import summary
from tqdm import tqdm

import pth_models

FLAGS = []

def running_metrics(preds, labels, tp, tn, fp, fn):
  tp += torch.sum((preds == 1) & (labels == 1)).item()
  tn += torch.sum((preds == 0) & (labels == 0)).item()
  fp += torch.sum((preds == 1) & (labels == 0)).item()
  fn += torch.sum((preds == 0) & (labels == 1)).item()
  precision = tp/(tp + fp + 0.0001)
  recall = tp/(tp + fn + 0.0001)
  F1 = (2 * precision * recall)/(precision + recall + 0.0001)
  acc = (tp+tn)/(tp+fp+tn+fn)
  sens = tp/(tp+fn + 0.0001)
  spec = tn/(tn+fp + 0.0001)
  return acc, sens, spec, F1, tp, tn, fp, fn 


class Trainer():
  def __init__(self, dataloaders, results_dir, model_type='resnet18', 
               n_epochs=25, input_size=1024, weights=0.5, profiler_mode=False):
    self.profiler_mode = profiler_mode
    self.model_type = model_type
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.n_epochs = 1 if profiler_mode else n_epochs
    self.input_size = input_size
    self.weights = weights
    self.dataloaders = dataloaders
    self.results_dir = results_dir
    for folder in [results_dir, os.path.join(results_dir, 'sample_imgs/')]:
      if not os.path.exists(folder): os.makedirs(folder)

  def set_parameter_requires_grad(self, model, mode='full_train', verbose=True):
    """ 
    Sets parameters requiring gradients for fine tuning a model or 
    full training

    Assumes that model consists of a base and a top module, such that 
    model.chidren() will have two items

    # Args
      model
      mode: one of 'full_train' or 'fine_tune'
    """
    
    if mode == 'fine_tune':
      def set_last_child_trainable():
        for param in model.parameters():
          param.requires_grad = False
        for param in children[-1].parameters():
          param.requires_grad = True

      children = list(model.children())
      if len(children) == 2:
        for param in children[0].parameters():
          param.requires_grad = False
        for param in children[1].parameters():
          param.requires_grad = True
      elif len(children) == 1:
        # used for mobilenet
        children = list(children[0].children())
        set_last_child_trainable()
      else:
        set_last_child_trainable()
    elif mode == 'full_train':
      for param in model.parameters():
        param.requires_grad = True
    if verbose:
      print('training gradients on {}/{} parameters'.format(
            len([p for p in model.parameters() if p.requires_grad == True]),
            len(list(model.parameters() )))
            )  
  
  def save_ckpt(self, model, ckpt_name):
    save_path = os.path.join(self.results_dir, ckpt_name)
    torch.save(model.state_dict(), save_path)

  def train_loop(self, model):
    start_time = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, self.n_epochs+1):
      print()
      if epoch < 0: #TODO back to 2
        self.set_parameter_requires_grad(model, mode='fine_tune')
      else:
        self.set_parameter_requires_grad(model, mode='full_train', 
                                         verbose=epoch==2)

      for phase in ['train', 'val']:       
        if phase == 'train':
          model.train() 
          if type(self.dataloaders['train']) is dict:
            dataloader = self.dataloaders['train'][epoch]
          else:
            dataloader = self.dataloaders['train']
        else:
          model.eval() 
          dataloader = self.dataloaders['val']

        phase_start_time = time.time()
        running_loss = 0.0
        batches_seen = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        batch_size = dataloader.dataset.batch_size
        n_batches = len(dataloader.dataset)//batch_size
        with tqdm(total=n_batches, unit=' batches',
                  bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|'
                             '{bar}|{postfix} [{elapsed}<{remaining}]') as pbar:
          pbar.set_description('{}: {}/{}'.format(phase, epoch, self.n_epochs))
          # save a random batch of images each epoch

          for i, batch in enumerate(dataloader):
            pbar.update(1)
            if len(batch) == 1:
              # dataloader batches are a list of 1 item, with that 
              # containing a list of 2 items, the first of which is 
              # images and the second is labels
              inputs = batch[0][0]
              labels = batch[0][1]
              batch_size = len(inputs) 
            else:
              inputs = batch[0]
              batch_size = len(inputs) 
              labels = batch[1].reshape(batch_size)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            if i in [1,2,3]: #img_save_index:
              save_path = os.path.join(self.results_dir, 'sample_imgs',
                                       'epoch{}-{}{}.jpg'.format(epoch, phase,
                                        i)) 
                                       
              torchvision.utils.save_image(inputs, save_path, nrow=4, 
                                           padding=4, normalize=True)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
              if self.model_type=='inception' and phase == 'train':
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = model(inputs)
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
              else:
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
              # torch.max() returns tensors for the max value and its position
              _, preds = torch.max(outputs, 1)

              if phase == 'train':
                loss.backward()
                self.optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            batches_seen += 1
            imgs_seen = batches_seen*batch_size
            acc, sens, spec, F1, tp, tn, fp, fn = running_metrics(preds, labels,
                                                  tp, tn, fp, fn)
            metrics = {'loss': round(running_loss/imgs_seen, 3),
                       'acc': round(acc, 3),
                       'sens': round(sens, 3),
                       'spec': round(spec, 3),
                       'F1': round(F1, 3),
                       } 
     
            pbar.set_postfix(metrics)  
        
        phase_time = round((time.time() - phase_start_time)/3600, 1)
         
        with open(os.path.join(self.results_dir, 'run_logs.txt'), 'a') as f:
          print('epoch {} - {} metrics ({} h):'.format(epoch, phase, 
                                                       phase_time), 
                metrics, file=f)
          if phase == 'val': print()

        # deep copy the model
        if phase == 'val' and metrics['acc'] > best_acc:
          best_acc = metrics['acc']
          best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'val':
          val_acc_history.append(metrics['acc'])

      self.save_ckpt(model, 'ckpt_epoch_{}.pth'.format(epoch))

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                                                        time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    self.save_ckpt(model, 'ckpt_best_val.pth')
    return model, val_acc_history


  def test_loop(self, model):
    start_time = time.time()

    model.eval() 

    phase_start_time = time.time()
    running_loss = 0.0
    batches_seen = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    batch_size = self.dataloaders['test'].dataset.batch_size
    n_batches = len(self.dataloaders['test'].dataset)//batch_size
    
    with tqdm(total=n_batches, unit=' batches',
              bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|'
                         '{bar}|{postfix} [{elapsed}<{remaining}]') as pbar:
      pbar.set_description('test')

      for i, batch in enumerate(self.dataloaders['test']):
        pbar.update(1)
        if len(batch) == 1:
          # dataloader batches are a list of 1 item, with that containing 
          # a list of 2 items, the first of which is images and the second 
          # is labels
          inputs = batch[0][0]
          labels = batch[0][1]
          batch_size = len(inputs) 
        else:
          inputs = batch[0]
          batch_size = len(inputs) 
          labels = batch[1].reshape(batch_size)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(False):
          outputs = model(inputs)
          loss = self.criterion(outputs, labels)
          # torch.max() returns tensors for the max value and its position
          # softmax does not need to be applied because the value of 
          # the prediction is not used
          _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        batches_seen += 1
        imgs_seen = batches_seen*batch_size
        acc, sens, spec, F1, tp, tn, fp, fn = running_metrics(preds, labels,
                                              tp, tn, fp, fn)
        metrics = {'loss': round(running_loss/imgs_seen, 3),
                   'acc': round(acc, 3),
                   'sens': round(sens, 3),
                   'spec': round(spec, 3),
                   'F1': round(F1, 3),
                   } 
        pbar.set_postfix(metrics)  
    
    phase_time = round((time.time() - phase_start_time)/3600, 1)
     
    with open(os.path.join(self.results_dir, 'run_logs.txt'), 'a') as f:
      print('test metrics ({} h):'.format(phase_time), 
            metrics, file=f)

    time_elapsed = time.time() - start_time
    print('testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                                                    time_elapsed % 60))
    return acc

  def train_main(self, optimizer='adam', print_summaries=False, 
                 print_params=False):
    model = pth_models.initialize_model(self.model_type, num_classes=2) 
    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)
    model = model.to(self.device)
    if print_summaries: 
      print(model) 
      summary(model, (3, self.input_size, self.input_size))
    optimizers_dict = {
      'sgdm': optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
      'adam': optim.Adam(model.parameters()),
      }
    self.optimizer = optimizers_dict[optimizer]
    # note that the CrossEntropyLoss expects raw, unnormalized scores 
    # (i.e. no softmax) and combines nn.LogSoftmax() and nn.NLLLoss() 
    # in one single class
    if self.weights==0.5:
      self.criterion = nn.CrossEntropyLoss().to(self.device)
    else:
      w = torch.Tensor([1-self.weights, self.weights])
      self.criterion = nn.CrossEntropyLoss(w).to(self.device)
 
    model, hist = self.train_loop(model)
    print(hist)
    if not self.profiler_mode:
      test_acc = self.test_loop(model)
      return test_acc


