# Introduction
This is a port of Apache MxNet demo for image classification in C++ ported to a GStreamer plugin. You can use it to annotate the frames of the input video or modify it to add a custom model.

This plugin demonstrates the following features:
* Interfacing GStreamer plugins with C++ code
* Interfacing Apache MxNet with GStreamer
* Using OpenCV to draw over GStreamer buffer

# Setting up the Docker instance

Here is how I set up docker (on mac):
```
	brew install docker docker-machine
	docker-machine create --driver virtualbox dev
```

Each time you want to work with docker (after a reboot or when you open a new terminal window), ensure the VM is started and docker environment is ready.
```
	docker-machine start dev
	eval $(docker-machine env dev)
```

If building for the first time or if you need to modify the plugin, re-package the archive. This is needed because Dockerfile COPY does not copy the
directory structure.
```
	./pack.sh
```

To build and run the image, use the following:
```
	docker build -t gstest .
	docker run -it gstest
```

To stop and remove modified containers it shall be possible to do:
```
	docker stop $(docker ps -a -q)
	docker rm $(docker ps -a -q)
```

# Using the plugin

Recognize one input JPEG image
```	
	gst-launch-1.0 multifilesrc location=/tmp/img2.jpg loop=true  ! jpegparse  !  avdec_mjpeg ! videoconvert ! video/x-raw,format=RGB ! mxnet ! fakesink
```

Combine input images into a sequence and recognize each of them
```
	gst-launch-1.0  multifilesrc  location=/tmp/img%d.jpg index=0 caps="image/jpeg,framerate=\(fraction\)1/1" loop=true  ! jpegparse  !  avdec_mjpeg ! videoconvert ! video/x-raw,format=RGB !  mxnet  ! fakesink  sync=true
```

Use the following command to capture the recording to a file to verify that recognition result message is written over the video stream:
```
	gst-launch-1.0  multifilesrc  location=/tmp/img%d.jpg index=0 caps="image/jpeg,framerate=\(fraction\)1/1" loop=true  ! jpegparse  !  avdec_mjpeg ! videoconvert ! video/x-raw,format=RGB !  mxnet  ! videoconvert ! theoraenc ! oggmux ! filesink location=/tmp/foo.ogg
```

# Expected Result
Observe a sequence of messages in GStreamer output
```
	Best Result: [ lynx, catamount] id = 287, accuracy = 0.31229183
	Best Result: [ boathouse] id = 449, accuracy = 0.86502564
	Best Result: [ beagle] id = 162, accuracy = 0.30743387
```

If you captured the output to a file, inspect it with a video player and ensure that each frame has the text in GREEN showing the recognized category and accuracy. See the screenshots below for an example.

![Screenshot 1](./screenshots/screenshot_1.png?raw=true "Cat")
![Screenshot 2](./screenshots/screenshot_2.png?raw=true "House")
