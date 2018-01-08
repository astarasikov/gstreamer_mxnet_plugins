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
RUN apt-get -y install \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	gir1.2-gst-plugins-base \
	gir1.2-gstreamer-1.0 \
	gstreamer-1.0 \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-bad-videoparsers \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-tools

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
#COPY simple_numpy.py /pygst/simple_numpy/python/

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

###############################################################################
# Install MxNet dependencies
###############################################################################
RUN apt-get -y install libopenblas-dev libopencv-dev
RUN apt-get -y install python3-dev python3-pip python3-numpy python3-nose python3-nose-timer python3-requests
RUN pip3 install opencv-python easydict

###############################################################################
# Compile MxNet with Python3
###############################################################################

# Unfortunately GCC consumes too much memory while compiling certain files
# so we have to limit it to one thread to ensure we don't run out of memory
# on Tegra TX2 with 8GB RAM
RUN \
	cd /pygst && \
	git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet --branch 0.11.0 && \
	cd mxnet && \
	make -j1 USE_BLAS=openblas USE_OPENCV=1 && \
	cd python && \
	python3 setup.py install

################################################################################
## Install mxnet-face
################################################################################
#RUN apt-get -y install wget unzip
#RUN \
#	cd /pygst && \
#	git clone https://github.com/tornadomeet/mxnet-face.git && \
#	cd mxnet-face && \
#	cd model && \
#	./get-models.sh
#
################################################################################
## Install our PyGst plugin that uses MxNet-Face via Python 3
################################################################################
#RUN mkdir /pygst/mxnet-face/detection3/
#COPY mxnet3.tar.gz /pygst/mxnet-face/
#COPY mxnet-face.zip /pygst/mxnet-face/detection3/
#RUN \
#	cd /pygst/mxnet-face/detection3/ && \
#	tar xvf ../mxnet3.tar.gz -C .. && \
#	unzip mxnet-face.zip -x /
#
#WORKDIR /pygst/mxnet-face/detection3/
#ENV PYTHONPATH=${PYTHONPATH}:/pygst/mxnet-face/detection3/
#ENV GST_PLUGIN_PATH=${GST_PLUGIN_PATH}:/pygst/mxnet-face/detection3/
#

###############################################################################
# Install Inception model
# https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-bn.md
###############################################################################
RUN \
	mkdir -p /pygst/mxnet_models/Inception && \
	cd /pygst/mxnet_models/Inception && \
	curl -o inception-bn.tar.gz http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz && \
	tar xvf inception-bn.tar.gz

###############################################################################
# Install GStreamer MxNet plugin
###############################################################################

COPY src_gst_mxnet_cpp.tar /pygst/
RUN \
	cd /pygst/ && \
	tar xvf src_gst_mxnet_cpp.tar && \
	cd gst_mxnet_cpp && \
	export LIBS="-lmxnet" && \
	export LDFLAGS="-L/pygst/mxnet/lib" && \
	export CFLAGS=-I/pygst/mxnet/include && \
	export CPPFLAGS=-I/pygst/mxnet/include && \
	./autogen.sh --prefix=/pygst/prefix && \
	./configure --prefix=/pygst/prefix && \
	make install

### MxNet Makefile not provide an install target
### So we will need to copy the .so to the prefix or export LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/pygst/mxnet/lib/

###############################################################################
# Install GStreamer libav plugin for the demo pipeline to decode MJPG
###############################################################################
RUN apt-get -y install gstreamer1.0-libav libavcodec-extra
COPY pictures_test/img0.jpg /tmp/
COPY pictures_test/img1.jpg /tmp/
COPY pictures_test/img2.jpg /tmp/
