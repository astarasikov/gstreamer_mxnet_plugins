This is a collection of GStreamer plugins which demonstrate
using Apache MxNet library from both Python and C++.

List of plugins:
* gst_python_numpy: a simple GStreamer plugin written in Python 3.
* gst_python_mxnet_face: a port of Mxnet-Face project to GStreamer and Python 3. Detects faces in the video stream and draws green boxes around them. This one is very slow because no performance optimization was performed.
* [WIP] gst_cpp_mxnet_recognize: this is a port of Apache C++ demo which recognizes (one) object in the input video feed.

Each project contains a Dockerfile so that you can test them in a VM without going through the complex build procedures.

Licensing:
* Some code is taken from Apache MxNet samples which is under the Apache license
* Plugins themselves can be redistributed under the GPLv2 license, but I'm considering a more liberal BSD/MIT license.
* Currently some pictures for demo were taken from various stock picture sites appearing in Google Search results. These need to be replaced with freely redistributable alternatives.


TODO:
* Replace demo pictures with some free alternatives and rebase to remove originals
* Update license (use MIT/WTFPL?)
