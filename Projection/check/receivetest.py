import argparse
import sys

import curio
import pynng

recieve_addr = "tcp://172.25.111.30:2001"

s2 = pynng.Sub0()
s2.subscribe("")
s2.listen(recieve_addr)
try:
    while True:
        msg = s2.recv_msg()
        source_addr = str(msg.pipe.remote_address)
        content = msg.bytes.decode()
        print('{} says: {}'.format(source_addr, content))
except KeyboardInterrupt:
    pass
