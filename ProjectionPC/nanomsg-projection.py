from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms





import argparse
import sys

import curio
import pynng

recieve_addr=""
send_addr=""


s1 = pynng.Pub0()
s1.publish(send_addr)
s1.dial(send_addr)



s2 = pynng.Sub0()
s2.subscribe("")
s2.listen(recieve_addr)
recieved_images= []



#  we load the model
model_path = EXPERIMENT_ARGS['model_path']

if os.path.getsize(model_path) < 1000000:
  raise ValueError("Pretrained model was unable to be downlaoded correctly!")

ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']

# update the training options
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False

opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')


async def createMorphing():
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_images = [img_transforms(image) for image in images]
    batched_images = torch.stack(transformed_images, dim=0)



async def send(data):
    """
    Send the data and image back to touchdesigner
    """
    img_bytes = data.bytes()
    await s1.asend(img_bytes)
    await curio.sleep(1)





async def main():
    # p = argparse.ArgumentParser(description=__doc__)
    # p.add_argument(
    #     'mode',
    #     help='Whether the socket should "listen" or "dial"',
    #     choices=['listen', 'dial'],
    # )
    # p.add_argument(
    #     'addr',
    #     help='Address to listen or dial; e.g. tcp://127.0.0.1:13134',
    # )
    # args = p.parse_args()

    with pynng.Sub0() as sub:
        sub.subscribe("")
        sub.dial(address)
        msg = await sub.arecv_msg()
        source_addr = str(msg.pipe.remote_address)
        content = msg.bytes.decode()
        recieved_images.append(content)
        print('{} says: {}'.format(source_addr, content))






if __name__ == '__main__':
    try:
        curio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
