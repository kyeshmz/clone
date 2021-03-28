import argparse
import sys

import curio
import pynng

s2 = pynng.Pair1(polyamorous=True, dial="tcp://127.0.0.1:13136")
s2.send(b'Hey man')
s2.close()
