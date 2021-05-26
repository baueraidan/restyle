from argparse import Namespace
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

import dlib
from scripts.align_faces_parallel import align_face

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
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

# Load pretrained model
model_path = 'pretrained_models/restyle_psp_ffhq_encode.pt'
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




image_path1 = 'faces/the_weekend.jpg'
image_path2 = 'faces/margot_robbie.jpg'
original_image1 = Image.open(image_path1).convert("RGB")
original_image2 = Image.open(image_path2).convert("RGB")

if experiment_type == 'cars_encode':
  original_image = original_image1.resize((192, 256))
  original_image = original_image2.resize((192, 256))
else:
  original_image = original_image1.resize((256, 256))
  original_image = original_image2.resize((192, 256))

def run_alignment(image_path):
  if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print('Downloading files for aligning face image...')
    os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
    print('Done.')
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 

if experiment_type in ['ffhq_encode', 'toonify']:
  input_image1 = run_alignment(image_path1)
  input_image2 = run_alignment(image_path2)
else:
  input_image1 = original_image1
  input_image2 = original_image2

input_image1.resize((256, 256))
input_image2.resize((256, 256))





img_transforms = EXPERIMENT_ARGS['transform']
transformed_image1 = img_transforms(input_image1)
transformed_image2 = img_transforms(input_image2)

def get_avg_image(net):
  avg_image = net(net.latent_avg.unsqueeze(0),
                input_code=True,
                randomize_noise=False,
                return_latents=False,
                average_code=True)[0]
  avg_image = avg_image.to('cuda').float().detach()
  if experiment_type == "cars_encode":
    avg_image = avg_image[:, 32:224, :]
  return avg_image

opts.n_iters_per_batch = 5
opts.resize_outputs = False  # generate outputs at full resolution

from utils.inference_utils import run_on_batch

with torch.no_grad():
  avg_image = get_avg_image(net)
  tic = time.time()
  result_batch1, result_latents1 = run_on_batch(transformed_image1.unsqueeze(0).cuda(), net, opts, avg_image)
  result_batch2, result_latents2 = run_on_batch(transformed_image2.unsqueeze(0).cuda(), net, opts, avg_image)
  toc = time.time()
  print('Inference took {:.4f} seconds.'.format(toc - tic))






if opts.dataset_type == "cars_encode":
    resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
else:
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

def get_coupled_results(result_batch, transformed_image):
  """
  Visualize output images from left to right (the input image is on the right)
  """
  result_tensors = result_batch[0]  # there's one image in our batch
  result_images = [tensor2im(result_tensors[iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
  input_im = tensor2im(transformed_image)
  res = np.array(result_images[0].resize(resize_amount))
  for idx, result in enumerate(result_images[1:]):
    res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
  res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
  res = Image.fromarray(res)
  return res

print('Result 1:')
#pprint.pprint(result_latents1[0].shape)
res1 = np.array(result_latents1[0])
pprint.pprint(res1)

print('Result 2:')
#pprint.pprint(result_latents1[0].shape)
res2 = np.array(result_latents2[0])
pprint.pprint(res2)

mean = (res1 + res2) * 0.5
print('Mean:')
pprint.pprint(mean.shape)
pprint.pprint(mean)

print('Codes:')
codes = mean[-1].reshape((1, 18, 512)).cuda()
pprint.pprint(codes.shape)
pprint.pprint([codes])

#result_lat = (np.array(result_latents1) + np.array(result_latents2)) * 0.5
#result_lat_tensor = 

# result_batch = (result_batch1 + result_batch2) * 0.5
# result_latents = (result_latents1 + result_latents2) * 0.5

# result_latents = (np.array(result_latents1[0]) + np.array(result_latents2[0])) * 0.5
# result_latents = result_latents[0].reshape((1, 18, 512))
# result_latents = torch.from_numpy(result_latents).cuda()
# pprint.pprint(result_latents.shape)
# pprint.pprint(result_latents)

# res = get_coupled_results(result_batch, transformed_image1)

# save image 
res.save(f'./{experiment_type}_results.jpg')
