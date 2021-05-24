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

print('Hello, world')

experiment_type = 'ffhq_encode' #@param ['ffhq_encode', 'cars_encode', 'church_encode', 'horse_encode', 'afhq_wild_encode', 'toonify']