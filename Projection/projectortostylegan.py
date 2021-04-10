# import modules
import argparse
import base64
import glob
import io
import json
import os
import re
import sys
import time
from argparse import Namespace
from io import BytesIO
from math import ceil

import curio
import cv2
import cv2 as cv
import dlib
import dnnlib
import dnnlib.tflib as tflib
import imageio
import IPython.display
import numpy as np
import PIL.Image
import pretrained_networks
import pynng
import scipy.ndimage
import torch
import torchvision as tv
import trio
from PIL import Image, ImageDraw
from pynng import Pair0

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from imutils import face_utils
from models.psp import pSp
from utils.common import tensor2im

print('starting')
print('starting tensorflow load')

# start stylegan configs
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

steps = 60  # 片側．生成される画像は，(steps-1) x 2 + 1
H = W = 512

SEEDs = [4336458, 222181]

# linspace
linspace = np.linspace(0, 1.0, steps)
# tmp = -1 * np.sort(linspace)[::-1]
# linspace = np.hstack((tmp[:-1], linspace))
print('linspace', linspace)
linspace = linspace.reshape(-1, 1, 1).astype(np.float32)

fromSeed = SEEDs[0]
toSeed = SEEDs[1]
seeds = [fromSeed, toSeed]
zs = generate_zs_from_seeds(seeds)

# end stylegan configs
print('starting pytorch loads')
transform = tv.transforms.Compose([
    tv.transforms.Resize((256, 256)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#  Load Pretrained Model
model_path = '../pretrained_models/psp_ffhq_encode.pt'
opts = torch.load(model_path, map_location='cuda:0')['opts']
opts['checkpoint_path'] = model_path
# if 'learn_in_w' not in opts: opts['learn_in_w'] = False
net = pSp(Namespace(**opts))
net.eval()
net.to('cuda:0')
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(
    './shape_predictor_68_face_landmarks.dat')
addr = ""

try:
    run_sync = trio.to_thread.run_sync
except AttributeError:
    # versions of trio prior to 0.12.0 used this method
    run_sync = trio.run_sync_in_worker_thread


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


async def image_align(src_file, lm, enable_padding=False):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    # lm = np.array(face_landmarks)
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Open image
    # img = src_file
    img = Image.open(src_file)

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform(img.size, Image.QUAD, (quad + 0.5).flatten(),
                        Image.BILINEAR)

    return img


async def recv_eternally(sock):
    while True:
        # global net
        # global transform
        # global face_detector
        # global face_predictor
        all_st = time.time()

        recv_msg = await sock.arecv_msg()
        recv_img_bytes = base64.b64decode(recv_msg.bytes.decode())
        print(type(recv_img_bytes))

        recv_bytes = io.BytesIO(recv_img_bytes)

        PIL_image = Image.open(recv_bytes)
        target_img = np.array(PIL_image)
        PIL_image.save('test3.jpg')

        img_path = 'test3.jpg'
        st = time.time()

        # Alignment
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        face = face_detector(img, 1)
        landmarks = face_predictor(img, face[0])
        landmarks = face_utils.shape_to_np(landmarks)
        img = await image_align(img_path, landmarks)

        img.save('aligned_test3.png')
        # continue
        # img = Image.open(img_path)

        ed = time.time()
        print(f'align {ed-st:.2f}', end=', ')
        # Embedding
        img = transform(img).unsqueeze(0)
        img, latents = net(img.to('cuda:0').float(),
                           return_latents=True,
                           randomize_noise=False)
        img = tensor2im(img[0])
        latents = latents.to('cpu').detach().numpy()
        print(f'embed {time.time()-ed:.2f}')

        # Save
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')

        latent_bytes = io.BytesIO()
        np.save(latent_bytes, latents)
        total_time = time.time() - all_st
        print(f'{total_time:.2f}, {total_time/len(img_path):.2f}')
        # print(typeof(img))
        img_64 = base64.b64encode(img_bytes.getvalue())
        print(type(img_bytes.getvalue()))
        print(latent_bytes)

        send_dict = {
            'ip': str(recv_msg.pipe.remote_address),
            'image': img_64,
            'latent': str(latent_bytes.getvalue())
        }
        send_dict_data = json.dumps(send_dict, indent=2).encode('utf-8')
        print('sending')
        for pipe in sock.pipes:
            await pipe.asend(send_dict_data)
            # await pipe.asend(img_64)
            # await pipe.asend(latents.tobytes())

        #  start of tensorlfow
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
        for img_idx, img in enumerate(imgs):
            img = PIL.Image.fromarray(img)
            img = img.resize((H, W), PIL.Image.ANTIALIAS)
            img.save('/results/img_' + str(100 + img_idx) + '.jpg')


async def main():

    td_addr = "tcp://172.25.111.30:5001"
    proj_addr = "tcp://172.25.157.35:5001"

    print('starting')

    model_path = 'pretrained_models/psp_ffhq_encode.pt'

    print("ready")

    with pynng.Pair1(polyamorous=True) as sock:
        async with trio.open_nursery() as n:

            def pre_connect_cb(pipe):
                addr = str(pipe.remote_address)
                print('~~~~got connection from {}'.format(addr))

            def post_remove_cb(pipe):
                addr = str(pipe.remote_address)
                print('~~~~goodbye for now from {}'.format(addr))

            sock.add_pre_pipe_connect_cb(pre_connect_cb)
            sock.add_post_pipe_remove_cb(post_remove_cb)
            sock.dial(td_addr)
            n.start_soon(recv_eternally, sock)


if __name__ == '__main__':
    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
