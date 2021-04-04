"""
Mini library used to read video data from an NDI source
"""

from ndi.lib import lib, ffi

from ndi.finder import create_ndi_finder
from ndi.receiver import create_receiver

# for typings
from ndi.finder import NDIFinder, NDISource


