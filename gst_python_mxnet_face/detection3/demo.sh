#!/usr/bin/env sh

export MXNET_ENGINE_TYPE=NaiveEngine
python3 -u mxnet_face_wrapper.py --img det.jpg --gpu 0 
