import io
import json
import pickle
import threading
import trio
import pynng
import multiprocessing
from PIL import Image
from pythonosc import udp_client

td_addr = "tcp://172.25.111.30:5001"
test_addr = "127.0.0.1"
proj_addr = "tcp://172.25.157.35:3001"

td1="ipc:///td/capture_img"
td2="ipc:///td2/capture_img2"

def write_images(images):
    imagefilename = './morphing_images/' + str(index) + '.png'
    image.save(imagefilename)
def write_align(image, index):
    imagefilename = './morphing_images/' +  str(index)+ '.png'
    image.save(imagefilename)

async def proj_main1(sock, client):
    while True:
        recv_msg_bytes = await sock.arecv_msg()
        recv_msg_pkl = pickle.loads(recv_msg_bytes.bytes)

        morphing_imgs = recv_msg_pkl.get('morphing_images')
        aligned_from = recv_msg_pkl.get('aligned_from')
        aligned_to = recv_msg_pkl.get('aligned_to')
        write_align(aligned_from, 1)
        write_align(aligned_to,2)

        p = multiprocessing.Pool()
        morph_images = p.map(write_images, imgs)
        morph_images = [entry for sublist in morph_images for entry in sublist]
        p.close()
        p.join()
        client.send_message("/recieve", 1)


async def proj_main1():
    while True:
        sock = pynng.Pair0(dial=td_addr)
        client = udp_client.SimpleUDPClient(test_addr, 8000)

        recv_msg_bytes = await sock.arecv_msg()
        recv_msg_pkl = pickle.loads(recv_msg_bytes.bytes)

        morphing_imgs = recv_msg_pkl.get('morphing_images')
        aligned_from = recv_msg_pkl.get('aligned_from')
        aligned_to = recv_msg_pkl.get('aligned_to')
        write_align(aligned_from, 1)
        write_align(aligned_to,2)

        p = multiprocessing.Pool()
        morph_images = p.map(write_images, imgs)
        morph_images = [entry for sublist in morph_images for entry in sublist]
        p.close()
        p.join()
        client.send_message("/recieve", 1)

async def proj_main2():
    while True:
        sock = pynng.Pair0(dial=td_addr)
        client = udp_client.SimpleUDPClient(test_addr, 8000)

        recv_msg_bytes = await sock.arecv_msg()
        recv_msg_pkl = pickle.loads(recv_msg_bytes.bytes)

        morphing_imgs = recv_msg_pkl.get('morphing_images')
        aligned_from = recv_msg_pkl.get('aligned_from')
        aligned_to = recv_msg_pkl.get('aligned_to')
        write_align(aligned_from, 1)
        write_align(aligned_to,2)

        p = multiprocessing.Pool()
        morph_images = p.map(write_images, imgs)
        morph_images = [entry for sublist in morph_images for entry in sublist]
        p.close()
        p.join()
        client.send_message("/recieve", 1)



async def main():
    client = udp_client.SimpleUDPClient(test_addr, 8000)
    with pynng.Pair1(polyamorous=True, dial=td_addr) as sock:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(recv_eternally, sock,client)
            nursery.start_soon(proj_main1)
            nursery.start_soon(proj_main2)

print('sending to ', test_addr)
print('port 8000')
print('listening to ', td_addr)
trio.run(main)
