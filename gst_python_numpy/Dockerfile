# use ubuntu zesty because it has GStreamer-1.0 version 1.10
# if older ubuntu is needed, we have two options:
# 1. compile gstreamer and all plugins from source
# 2. add zesty repositories but set them to lower apt priority
# so that all packages are taken from our default release
# except the ones we explicitel install from a different version
FROM ubuntu:zesty

# we're compiling for python3
# however to use GObject in Ubuntu/Debian we need to install python-gi-dev
# which depends on python2 even if we only use python3

###############################################################################
# Install dependencies: GStreamer-1.0 and python development packages
###############################################################################

RUN apt-get -y update
RUN apt-get -y install git build-essential autoconf autogen libtool pkg-config
RUN apt-get -y install libgstreamer1.0-dev gir1.2-gst-plugins-base gir1.2-gstreamer-1.0 gstreamer-1.0 gstreamer1.0-tools
RUN apt-get -y install python3-dev python3-gi python-gi-dev python3-gst-1.0 python3-gst-1.0-dbg
RUN apt-get -y install python3-numpy

###############################################################################
# environment variables for running GStreamer with our custom plugin
###############################################################################
ENV PYTHON=python3
ENV GST_PLUGIN_PATH=/pygst/prefix/lib/gstreamer-1.0/:/pygst/gst-python/examples/plugins:/pygst/simple_numpy/

RUN mkdir -p /pygst/prefix /pygst/simple_numpy
WORKDIR /pygst
# python plugins must be in the "python" dir
# but GST_PLUGIN_PATH must be one level up
COPY simple_numpy.py /pygst/simple_numpy/python/

###############################################################################
# Download, compile and install gst-python
###############################################################################
RUN \
	cd /pygst && \
	git clone https://github.com/GStreamer/gst-python.git && \
	cd gst-python && \
	git checkout origin/1.10 && \
	./autogen.sh --prefix=/pygst/prefix --with-libpython-dir=/usr/lib/x86_64-linux-gnu && \
	make install

#copy simple_numpy.py
