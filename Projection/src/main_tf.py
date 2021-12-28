import argparse
import base64
import glob
import io
import multiprocessing
import os
import pickle
import re
import sys
import time
import typing
from math import ceil
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

import dlib
import IPython.display
import numpy as np
import pynng
import scipy.ndimage
import torch
import asyncio
import torchvision as tv
import trio
import socket
from pythonosc import dispatcher
from pythonosc import osc_server
from PIL import Image, ImageDraw, ImageOps
from imutils import face_utils

# we assume the following directory format
# Projection
# 11_src
# --22_lib
# ----ps2p files
# ----stylegan2 files

from lib.ps2p.models.psp import pSp

import dnnlib
import dnnlib.tflib as tflib





async def pil_to_numpy(image: Image):
    image = image.convert("RGB")
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    return image


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def generate_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        zs.append(z)
    return zs


async def generate_images_from_ws(dlatents):
    imgs = []
    for row, dlatent in enumerate(dlatents):
        img = Gs.components.synthesis.run(dlatent, **Gs_kwargs)
        imgs.append(img[0])
    return imgs


gdrive_urls = {
    'gdrive:networks/stylegan2-ffhq-config-f.pkl':                          'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
}

def get_path_or_url(path_or_gdrive_path):
    return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)
_cached_networks = dict()

def load_networks(path_or_gdrive_path):
    path_or_url = get_path_or_url(path_or_gdrive_path)
    if path_or_url in _cached_networks:
        return _cached_networks[path_or_url]

    if dnnlib.util.is_url(path_or_url):
        stream = dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
    else:
        stream = open(path_or_url, 'rb')

    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path_or_url] = G, D, Gs
    return G, D, Gs


# def load_networks(path_or_gdrive_path):
#     stream = open(path_or_gdrive_path, 'rb')
#     # tf session is created inside of here
#     tflib.init_tf()
#     with stream:
#         G, D, Gs = pickle.load(stream, encoding='latin1')
    
#     return G, D, Gs
# this didnt work or rather didnt release memory
# def generate_images_from_ws_array(dlatents):
#     imgs = []
#     # for row, dlatent in enumerate(dlatents):
#     #     img = Gs.components.synthesis.run(dlatent, **Gs_kwargs)
#     #     imgs.append(img[0])

#     img = Gs.components.synthesis.run(dlatents, **Gs_kwargs)
#     imgs.append(img)
#     return imgs


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


def resize(img):
    imglist = []
    img = Image.fromarray(img)
    img = img.resize((H, W), Image.ANTIALIAS)
    imglist.append(img)
    return imglist

incoming_ip = ''
steps = 0
def setSettings(unused_addr, *args):
    incoming_ip = args[0]
    steps = args[1]
    print("args1 ",args[0])
    print("args2 ",args[1])
    asyncio.run(main("tcp://"+incoming_ip, incoming_ip, steps))


async def main(pairip,incoming_ip, steps):
    with pynng.Pair1(polyamorous=True) as sock:
        print(pairip)
        print(type(pairip))
        sock.dial(pairip)
        recv_msg = await sock.arecv_msg()
        recieved_time = time.time()
        print('recieved')
        recv_pkl = pickle.loads(recv_msg.bytes)
        from_img_msg = recv_pkl.get('from')
        to_img_msg = recv_pkl.get('to')

        from_img_decode = base64.b64decode(from_img_msg)
        from_img_bytes = io.BytesIO(from_img_decode)
        from_PIL = Image.open(from_img_bytes)
        from_NP = np.array(from_PIL)

        to_img_decode = base64.b64decode(to_img_msg)
        to_img_bytes = io.BytesIO(to_img_decode)
        to_PIL = Image.open(to_img_bytes)
        to_NP = np.array(to_PIL)

        st = time.time()
         # Alignment
        try:
            from_face = face_detector(from_NP, 1)
            if (len(from_face) == 0 ):
                return
            from_landmarks = face_predictor(from_NP, from_face[0])
            from_landmarks = face_utils.shape_to_np(from_landmarks)
            from_alignimg = await image_align(from_PIL, from_landmarks)
          
            from_alignimgnp = await pil_to_numpy(from_alignimg)
       

            to_face = face_detector(to_NP, 1)
            to_landmarks = face_predictor(to_NP, to_face[0])
            to_landmarks = face_utils.shape_to_np(to_landmarks)
            to_alignimg = await image_align(to_PIL, to_landmarks)
            to_alignimgnp = await pil_to_numpy(to_alignimg)
       

            ed = time.time()
            print(f'align {ed-st:.2f}', end=', ')
            # Embedding
            from_embedimg = transform(from_alignimg).unsqueeze(0)
            from_projimg, from_latents = net(
                from_embedimg.to('cuda:0').float(),
                return_latents=True,
                randomize_noise=False)
            from_latents = from_latents.to('cpu').detach().numpy()

            to_embedimg = transform(to_alignimg).unsqueeze(0)
            to_projimg, to_latents = net(to_embedimg.to('cuda:0').float(),
                                         return_latents=True,
                                         randomize_noise=False)
            to_latents = to_latents.to('cpu').detach().numpy()
            embed_time = time.time()
            print(f'embed {embed_time-ed:.2f}')

            #  start of tensorflow 
            # W space vector
            dlatent_from = from_latents
            dlatent_to = to_latents

            dlatent_morph = (dlatent_to - dlatent_from)
            
            # linspace = np.linspace(0, 1.0, steps)
            # linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
           

            edited_dlatents = dlatent_from + linspace * dlatent_morph
            edited_dlatents = edited_dlatents.reshape(STEPS, 1, 18, 512)
            # for making faster, but still gpu problem
            # edited_dlatents = edited_dlatents.reshape(steps, 18, 512)
   

            imgs = await generate_images_from_ws(edited_dlatents)
            # 120x (1024,1024,3)
            generate_time = time.time()

            print(f'generate {generate_time-embed_time:.2f}')
            morph_images = []

            p = multiprocessing.Pool(16)
            morph_images = p.map(resize, imgs)
            morph_images = [
                entry for sublist in morph_images for entry in sublist
            ]
            p.close()
            p.join()

            print(f'creation {time.time()-generate_time:.2f}')
            # sending
            send_data = {
                "aligned_from": from_alignimgnp,
                "aligned_to": to_alignimgnp,
                "morphing_images": morph_images
            }
            send_data_pkl = pickle.dumps(send_data)
            print('sending')
            print('morph length', len(morph_images))
            sock.send(send_data_pkl)
            # for pipe in sock.pipes:
            # await pipe.asend(send_data_pkl)
            print('done sending')
            print(f'total time {time.time()-recieved_time:.2f}')
            await asyncio.sleep(1)
        except:
            client = SimpleUDPClient(incoming_ip, 4000)
            client.send_message("/error", 1)
            print('error')
            await asyncio.sleep(1)




# start stylegan configs
#------------------

network_pkl = "lib/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
print('Loading networks from "%s"...' % network_pkl)


_, _, Gs = load_networks(network_pkl)
noise_vars = [
    var for name, var in Gs.components.synthesis.vars.items()
    if name.startswith('noise')
]
print(Gs)
print('test')
Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                  nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
print(Gs_kwargs)

# max is 20
batchsize = 15
Gs_kwargs.minibatch_size = batchsize

STEPS = 120
H = W = 256

SEEDs = [4336458, 222181]

# linspace
linspace = np.linspace(0, 1.0, STEPS)
print('linspace', linspace)
linspace = linspace.reshape(-1, 1, 1).astype(np.float32)

fromSeed = SEEDs[0]
toSeed = SEEDs[1]
seeds = [fromSeed, toSeed]
zs = generate_zs_from_seeds(seeds)

print('starting pytorch loads')
transform = tv.transforms.Compose([
    tv.transforms.Resize((256, 256)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# end stylegan configs
#-------------------

#  Load Pretrained Model
model_path = './lib/ps2p/pretrained_models/psp_ffhq_encode.pt'
opts = torch.load(model_path, map_location='cuda:0')['opts']
opts['checkpoint_path'] = model_path
# if 'learn_in_w' not in opts: opts['learn_in_w'] = False
net = pSp(argparse.Namespace(**opts))
net.eval()
net.to('cuda:0')
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(
    './lib/dlib/shape_predictor_68_face_landmarks.dat')



#  we assume the following
#  first: ip
#  second: amount
#  third: fromimg
#  fourth: toimg
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/create", setSettings)


ip = "0.0.0.0"
port = 5000

# async def init_main():
#     server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())
#     transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
#     while True:
#         await main()
 
# asyncio.run(init_main())



# server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())

server = osc_server.ThreadingOSCUDPServer(
    (ip, port), dispatcher)
server.serve_forever()
#print("Serving on {}".format(server.server_address))
# server.serve()


