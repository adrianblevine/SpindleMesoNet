import torch
import torch.nn as nn
import torchvision
from torchvision import models

if int(torchvision.__version__.split('.')[1]) <= 3:
  from new_models import mobilenet_v2, shufflenet_v2_x1_0 
  from new_models.resnet import resnext50_32x4d
else:
  from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, \
                                 resnext50_32x4d, mnasnet1_0


def initialize_model(model_type, num_classes=2, feature_extract=False, use_pretrained=True):
  """ Initialize these variables which will be set in this if statement. 
  Each of these variables is model specific.
  """
  model = None

  if model_type == "resnet18":
    """ Resnet18
    the main challenge here is handling dimensions in the transition from 
    the lastpooling layer to the linear layer the current 
    torchvision implementation uses nn.AdaptiveAvgPool2d((1,1)), while 
    previous versions used AvgPool2d(kernel_size=7, stride=1, padding=0)
    in the standard model, the input to the pooling layer will be of shape:
    [batch_size, 512, input_size/32, input_size/32]
    recall that out_dim = (in_dim + 2*padding - kernel_size)/stride + 1,
    so with padding=0 and stride=1, out_dim = in_dim - kernel_size + 1
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)

  elif model_type == 'resnet34':
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)

  elif model_type == 'resnet50':
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)

  elif model_type == "resnet18_custom":
    model = resnets.resnet18(pretrained=True, input_shape=1024)

  elif model_type == "mobilenet":
    model = mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2) 

  elif model_type == "resnext":
    model = resnext50_32x4d(pretrained=True)

  elif model_type == 'shufflenet':
    model = shufflenet_v2_x1_0(pretrained=True)

  elif model_type == "alexnet":
    """ Alexnet
    """
    model = models.alexnet(pretrained=use_pretrained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)

  elif model_type == "vgg":
    """ VGG11_bn
    """
    model = models.vgg11_bn(pretrained=use_pretrained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)

  elif model_type == "squeezenet":
    """ Squeezenet
    """
    model = models.squeezenet1_0(pretrained=use_pretrained)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), 
                                           stride=(1,1))
    model.num_classes = num_classes

  elif model_type == "densenet121":
    """ Densenet
    """
    model = models.densenet121(pretrained=use_pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes) 

  elif model_type == "inception":
    """ Inception v3 
    Be careful, expects (299,299) sized images and has auxiliary output
    """
    model = models.inception_v3(pretrained=use_pretrained)
    # Handle the auxilary net
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)

  elif model_type == 'mnasnet':
    model = mnasnet1_0(pretrained=False, progress=True)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                     nn.Linear(1280, num_classes))

  else:
    print("Invalid model name, exiting...")
    exit()
  
  return model



