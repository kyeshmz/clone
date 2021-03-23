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




async def createMorphing()


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
        print('{} says: {}'.format(source_addr, content))






if __name__ == '__main__':
    try:
        curio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
