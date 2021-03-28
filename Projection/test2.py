# import modules
import argparse
import base64
import glob
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
from imutils import face_utils
from models.psp import pSp
from PIL import Image
from utils.common import tensor2im


def image_align(src_file, lm, enable_padding=False):
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
    img = src_file
    # img = Image.open(src_file)

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


def NdarrayToBase64(dst):
    result, dst_data = cv2.imencode('.jpg', dst)
    dst_base64 = base64.b64encode(dst_data)
    # dst_data = dst_data.tobytes()

    return dst_base64


if __name__ == '__main__':

    addr = 'tcp://172.25.157.35:5321'

    input_dir = 'myimages/raw_old/'
    output_dir = 'myimages/result/'
    model_path = 'pretrained_models/psp_ffhq_encode.pt'
    device = 'cuda:0'
    transform = tv.transforms.Compose([
        tv.transforms.Resize((256, 256)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    #  Load Pretrained Model
    opts = torch.load(model_path, map_location=device)['opts']
    opts['checkpoint_path'] = model_path
    # if 'learn_in_w' not in opts: opts['learn_in_w'] = False
    net = pSp(Namespace(**opts))
    net.eval()
    net.to(device)
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(
        './shape_predictor_68_face_landmarks.dat')

    all_st = time.time()

    img_paths = glob.glob(input_dir + '*')
    while True:
        with pynng.Pair0(dial=addr) as sock:
            recv_img_bytes = sock.recv()

            img_binary = base64.b64decode(recv_img_bytes)
            recv_img = np.frombuffer(img_binary, dtype=np.uint8)

            target_img = cv2.imdecode(recv_img, cv2.IMREAD_COLOR)

            PIL_image = Image.fromarray(np.uint8(target_img)).convert('RGB')

            print(type(target_img))
            print(target_img.shape)
            print(target_img.dtype)
            st = time.time()

            # Alignment

            img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            face = face_detector(img, 1)
            landmarks = face_predictor(img, face[0])
            landmarks = face_utils.shape_to_np(landmarks)
            img = image_align(PIL_image, landmarks)
            img.save(f'{output_dir}aligned_test2.png')
            # continue
            # img = Image.open(img_path)

            ed = time.time()
            print(f'align {ed-st:.2f}', end=', ')

            # Embedding
            img = transform(img).unsqueeze(0)
            img, latents = net(img.to(device).float(),
                               return_latents=True,
                               randomize_noise=False)
            img = tensor2im(img[0])
            latents = latents.to('cpu').detach().numpy()
            print(f'embed {time.time()-ed:.2f}')

            # Save
            img.save(f'{output_dir}embedded_test2.png')
            np.save(f'{output_dir}embedded_test2.npy', latents)
            # break

            total_time = time.time() - all_st
            print(f'{total_time:.2f}, {total_time/len(image_path):.2f}')
