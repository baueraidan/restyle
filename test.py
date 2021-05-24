from argparse import Namespace
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils.common import tensor2im
from models.psp import pSp
from models.e4e import e4e

experiment_type = 'ffhq_encode' #@param ['ffhq_encode', 'cars_encode', 'church_encode', 'horse_encode', 'afhq_wild_encode', 'toonify']

EXPERIMENT_DATA_ARGS = {
  "ffhq_encode": {
    "model_path": "pretrained_models/restyle_psp_ffhq_encode.pt",
    "image_path": "notebooks/images/face_img.jpg",
    "transform": transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  },
  "cars_encode": {
    "model_path": "pretrained_models/restyle_psp_cars_encode.pt",
    "image_path": "notebooks/images/car_img.jpg",
    "transform": transforms.Compose([
      transforms.Resize((192, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  },
  "church_encode": {
    "model_path": "pretrained_models/restyle_psp_church_encode.pt",
    "image_path": "notebooks/images/church_img.jpg",
    "transform": transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  },
  "horse_encode": {
    "model_path": "pretrained_models/restyle_e4e_horse_encode.pt",
    "image_path": "notebooks/images/horse_img.jpg",
    "transform": transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  },
  "afhq_wild_encode": {
    "model_path": "pretrained_models/restyle_psp_afhq_wild_encode.pt",
    "image_path": "notebooks/images/afhq_wild_img.jpg",
    "transform": transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  },
  "toonify": {
    "model_path": "pretrained_models/restyle_psp_toonify.pt",
    "image_path": "notebooks/images/toonify_img.jpg",
    "transform": transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

# Load pretrained model
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
pprint.pprint(opts)
# Update the training options
opts['checkpoint_path'] = model_path

opts = Namespace(**opts)
if experiment_type == 'horse_encode': 
  net = e4e(opts)
else:
  net = pSp(opts)
    
net.eval()
net.cuda()
print('Model successfully loaded!')




image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
original_image = Image.open(image_path).convert("RGB")