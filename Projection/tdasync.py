"""
Demonstrate how to use a pair1 socket.

Pair1 sockets are similar to pair0 sockets.  The difference is that while pair0
supports only a single connection, pair1 sockets support _n_ one-to-one
connections.

This program demonstrates how to use pair1 sockets.  The key differentiator is
that with pair1 sockets, you must always specify the *pipe* that you want to
use for the message.

To use this program, you must start several nodes.  One node will be the
listener, and all other nodes will be dialers.  In one terminal, you must start
a listener:

    python pair1_async.py listen tcp://127.0.0.1:12345

And in as many separate terminals as you would like, start some dialers:

    # run in as many separate windows as you wish
    python pair1_async.py dial tcp://127.0.0.1:12345

Whenever you type into the dialer processes, whatever you type is received on
the listening process.  Whatever you type into the listening process is sent to
*all* the dialing processes.

"""

# import modules
import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from argparse import Namespace

import curio
import cv2
import cv2 as cv
import dlib
import numpy as np
import pynng
import scipy.ndimage
import torch
import torchvision as tv
import trio
from PIL import Image
from pynng import Pair0

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from imutils import face_utils
from models.psp import pSp
from utils.common import tensor2im

print('starting')
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
