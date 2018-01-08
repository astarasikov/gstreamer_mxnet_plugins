#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# simple_numpy.py
# 2017 Alexander Tarasikov <alexander.tarasikov@gmail.com>
#
# Simple example of processing buffers using NumPy
# Unfortunately it involves some hacks
#
# You can run the example from the source doing from gst-python/:
#
#  $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/plugin:$PWD/examples/plugins
#  $ GST_DEBUG=python:4 gst-launch-1.0 fakesrc num-buffers=10 ! simple_numpy ! fakesink

###############################################################################
# GStreamer Python bindings
###############################################################################
import gi
gi.require_version('GstBase', '1.0')

from gi.repository import Gst, GObject, GstBase
Gst.init(None)

###############################################################################
# Use NumPy to manipulate raw C pointers
###############################################################################
import numpy as np

###############################################################################
# Dependencies for using GStreamer C library
###############################################################################
import ctypes
import platform
from os.path import dirname

###############################################################################
# Hack
# based on https://github.com/stb-tester/stb-tester/blob/master/_stbt/gst_hacks.py
###############################################################################
# Here we are using ctypes to call `gst_buffer_map` and `gst_buffer_unmap`
# because PyGObject does not properly expose struct GstMapInfo (see
# [bz #678663]).  Apparently this is fixed upstream but we are still awaiting
# an upstream release (Mar 2014).  Hopefully this can be removed in the future.

_GST_PADDING = 4  # From gstconfig.h


# From struct GstMapInfo in gstreamer/gst/gstmemory.h:
class _GstMapInfo(ctypes.Structure):
    _fields_ = [("memory", ctypes.c_void_p),   # GstMemory *memory
                ("flags", ctypes.c_int),       # GstMapFlags flags
                ("data", ctypes.POINTER(ctypes.c_byte)),    # guint8 *data
                ("size", ctypes.c_size_t),     # gsize size
                ("maxsize", ctypes.c_size_t),  # gsize maxsize
                ("user_data", ctypes.c_void_p * 4),     # gpointer user_data[4]
                # gpointer _gst_reserved[GST_PADDING]:
                ("_gst_reserved", ctypes.c_void_p * _GST_PADDING)]

_GstMapInfo_p = ctypes.POINTER(_GstMapInfo)

if platform.system() == "Darwin":
    _libgst = ctypes.CDLL(dirname(Gst.__path__) + "/../libgstreamer-1.0.dylib")
else:
    _libgst = ctypes.CDLL("libgstreamer-1.0.so.0")
_libgst.gst_buffer_map.argtypes = [ctypes.c_void_p, _GstMapInfo_p, ctypes.c_int]
_libgst.gst_buffer_map.restype = ctypes.c_int

_libgst.gst_buffer_unmap.argtypes = [ctypes.c_void_p, _GstMapInfo_p]
_libgst.gst_buffer_unmap.restype = None

_libgst.gst_buffer_get_size.argtypes = [ctypes.c_void_p]
_libgst.gst_buffer_get_size.restype = ctypes.c_size_t

###############################################################################
# Actual implementation of the plugin
###############################################################################

class SimpleNumpy(GstBase.BaseTransform):
    __gstmetadata__ = ('SimpleNumpy Python','Transform', \
                      'Simple processing using NumPy', 'Alexander Tarasikov')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                           Gst.PadDirection.SRC,
                                           Gst.PadPresence.ALWAYS,
                                           Gst.Caps.new_any()),
                       Gst.PadTemplate.new("sink",
                                           Gst.PadDirection.SINK,
                                           Gst.PadPresence.ALWAYS,
                                           Gst.Caps.new_any()))

# The macro "gst_buffer_make_writable" is not available
# because C macros are not exported into GIR bindings.
# Same for ref/unref functions.
#
# What we can do from python is to temporarily modify
# the refcount of the buffer to 1 to allow it to be mapped
# as writable. This should be safe because GObject
# reference counting is not thread-safe which means
# that normally other threads are not using the buffer concurrently.
#
# Unfortunately this still does not allow us to modify
# the buffer from python because the Gst.MapInfo "data"
# field contains the reference to the Python "bytes"
# type which is immutable. So we need to call into C libraries
# to obtain the raw pointer from which we can construct
# a NumPy object
#
# It looks like it's still not completely fixed upstream
# and maybe for some precompiled binary plugins we will
# need to use non-upstream build anyway
    def do_transform_ip(self, buffer_out):
        Gst.info("timestamp(buffer):%s" % (Gst.TIME_ARGS(buffer_out.pts)))
        
        # Get Frame dimensions
        config = buffer_out.pool.get_config()
        caps = config['caps']
        struct = caps.get_structure(0)
        (ok, width) = struct.get_int('width')
        if not ok:
            raise RuntimeError("Failed to get width")
        
        (ok, height) = struct.get_int('height')
        if not ok:
            raise RuntimeError("Failed to get height")

        mo = buffer_out.mini_object
        saved_refcount = mo.refcount
        mo.refcount = 1

        # for GObject instances, hash() returns the pointer to the C struct
        pbuffer = hash(buffer_out)
        mapping = _GstMapInfo()
        success = _libgst.gst_buffer_map(pbuffer, mapping, Gst.MapFlags.WRITE)
        if not success:
            raise RuntimeError("Could not map buffer")
        else:
            ctypes_region = ctypes.cast(mapping.data,
                    ctypes.POINTER(ctypes.c_byte * mapping.size))
            raw_ptr = ctypes_region.contents

            # create ctypes array from the raw pointer
            ptr = (ctypes.c_byte * mapping.size).from_address(ctypes.addressof(raw_ptr))

            # cast array to uint32 to work with RGBA/BGRx data
            # FIXME: hardcode caps to only allow 4-byte modes?
            ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            np_arr = np.ctypeslib.as_array(ptr, shape=(height, width))

            # this is fast, just one call into numpy and proably vectorized in C
            # if image is BGRx, then this will erase the red channel
            np_arr &= 0xff00ffff

            # invert colors for top quarter rows
            for i in range(height // 4):
                np_arr[i] ^= 0x00ffffff

            # this is SLOOOOW, multiple calls into numpy for each element
            # however, if performance is not a concern
            #(for example, processing video offline), you can add
            # sync=false to the last pipeline element (sink)
            # to avoid the "gst_base_sink_is_too_late" error
            #for i in range(height):
            #    for j in range(width):
            #        np_arr[i][j] //= 2

            _libgst.gst_buffer_unmap(pbuffer, mapping)

        mo.refcount = saved_refcount
        return Gst.FlowReturn.OK

GObject.type_register(SimpleNumpy)
__gstelementfactory__ = ("simple_numpy", Gst.Rank.NONE, SimpleNumpy)
