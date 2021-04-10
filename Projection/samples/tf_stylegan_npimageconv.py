import argparse
import os
import re
import sys
from io import BytesIO
from math import ceil

import dnnlib
import dnnlib.tflib as tflib
import imageio
import IPython.display
import numpy as np
import PIL.Image
import pretrained_networks
import tensorflow as tf
from PIL import Image, ImageDraw

#  code used from https://colab.research.google.com/drive/11shCb5N-01Rkl5GhnCtBKX0pdw3FZiuD?usp=sharing#scrollTo=BQIhdSRcXC-Q
network_pkl = "networks/stylegan2-ffhq-config-f.pkl"

print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [
    var for name, var in Gs.components.synthesis.vars.items()
    if name.startswith('noise')
]

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                  nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False


def generate_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        zs.append(z)
    return zs


def generate_images_from_ws(dlatents):
    imgs = []
    for row, dlatent in enumerate(dlatents):
        img = Gs.components.synthesis.run(dlatent, **Gs_kwargs)
        imgs.append(img[0])
    return imgs


def create_image_grid(imgs, size_=256):
    canvas = PIL.Image.new('RGB', (size_ * len(imgs), size_), 'white')
    for idx, img in enumerate(imgs):
        img = PIL.Image.fromarray(img)
        img = img.resize((size_, size_), PIL.Image.ANTIALIAS)
        canvas.paste(img, (size_ * idx, 0))
    return canvas


def createImageGrid(images, scale=0.25, rows=1):
    w, h = images[0].size
    w = int(w * scale)
    h = int(h * scale)
    height = rows * h
    cols = ceil(len(images) / rows)
    width = cols * w
    canvas = PIL.Image.new('RGBA', (width, height), 'white')
    for i, img in enumerate(images):
        img = img.resize((w, h), PIL.Image.ANTIALIAS)
        canvas.paste(img, (w * (i % cols), h * (i // cols)))
    return canvas


steps = 60  # 片側．生成される画像は，(steps-1) x 2 + 1
H = W = 512

SEEDs = [4336458, 222181]

# linspace
linspace = np.linspace(0, 1.0, steps)
# tmp = -1 * np.sort(linspace)[::-1]
# linspace = np.hstack((tmp[:-1], linspace))
print('linspace', linspace)
linspace = linspace.reshape(-1, 1, 1).astype(np.float32)

me_dlatent = np.load('npy/1shimizu_01.npy')
he_dlatent = np.load('npy/kasa1_01.npy')

# morphing start

fromSeed = SEEDs[0]
toSeed = SEEDs[1]
seeds = [fromSeed, toSeed]
zs = generate_zs_from_seeds(seeds)

#W space vector
dlatent_from = me_dlatent
print(dlatent_from.shape)
dlatent_to = he_dlatent
#Gs.components.mapping.run(zs[1], None)
print(dlatent_to.shape)

dlatent_morph = (dlatent_to - dlatent_from)
# print( np.linalg.norm(dlatent_morph))

edited_dlatents = dlatent_from + linspace * dlatent_morph
edited_dlatents = edited_dlatents.reshape(steps, 1, 18, 512)
print(edited_dlatents.shape)
imgs = generate_images_from_ws(edited_dlatents)
# print( len(imgs) )
# os.mkdir('./results')

for img_idx, img in enumerate(imgs):
    img = PIL.Image.fromarray(img)
    img = img.resize((H, W), PIL.Image.ANTIALIAS)
    img.save('results/img_' + str(100 + img_idx) + '.jpg')
