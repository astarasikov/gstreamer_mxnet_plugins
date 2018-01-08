/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 */

//
//  File: image-classification-predict.cpp
//  This is a simple predictor which shows
//  how to use c api for image classfication
//  It uses opencv for image reading
//  Created by liuxiao on 12/9/15.
//  Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
//  Home Page: www.liuxiao.org
//  E-mail: liuxiao@foxmail.com
//

#include <stdio.h>

// Path for c_predict_api
#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

const mx_float DEFAULT_MEAN = 117.0;

// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};

void GetImageFile(void *gst_data,
				unsigned gst_width,
				unsigned gst_height,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori(gst_width, gst_height, CV_8UC3, gst_data);

    if (im_ori.empty()) {
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
               mean_data++;
            }
            if (channels > 1) {
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(std::string synset_file) {
    std::ifstream fi(synset_file.c_str());

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while ( fi >> synset ) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        //printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if ( data[i] > best_accuracy ) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
    synset[best_idx].c_str(), best_idx, best_accuracy);
}

enum {
	MODEL_WIDTH= 224,
	MODEL_HEIGHT = 224,
	MODEL_CHANNELS = 3,
    MODEL_IMAGE_SIZE = MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS,
};

//TODO: move into (context-like) struct
static PredictorHandle pred_hnd = 0;
static NDListHandle nd_hnd = 0;
static const mx_float* nd_data = NULL;
static std::vector<std::string> synset;
static std::vector<mx_float> image_data = std::vector<mx_float>(MODEL_IMAGE_SIZE);

static std::vector<float> mxnet_out_data(1000);

/*
 * TODO: accept parameters, return struct
 */
int gst_mxnet_init_network(void)
{
    // Models path for your model, you have to modify it
    std::string json_file = "/pygst/mxnet_models/Inception/Inception-BN-symbol.json";
    std::string param_file = "/pygst/mxnet_models/Inception/Inception-BN-0126.params";
    std::string synset_file = "/pygst/mxnet_models/Inception/synset.txt";
    std::string nd_file = "/pygst/mxnet_models/Inception/mean_224.nd";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;

    // Image size and channels
    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(MODEL_CHANNELS),
                                        static_cast<mx_uint>(MODEL_HEIGHT),
                                        static_cast<mx_uint>(MODEL_WIDTH)};

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }

    // Create Predictor
    MXPredCreate((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    assert(pred_hnd);

    // Read Mean Data
    BufferFile nd_buf(nd_file);

    if (nd_buf.GetLength() > 0) {
        mx_uint nd_index = 0;
        mx_uint nd_len;
        const mx_uint* nd_shape = 0;
        const char* nd_key = 0;
        mx_uint nd_ndim = 0;

        MXNDListCreate((const char*)nd_buf.GetBuffer(),
                   nd_buf.GetLength(),
                   &nd_hnd, &nd_len);

        MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);
    }

	// Synset path for your model, you have to modify it
    synset = LoadSynset(synset_file);
	return 0;
}

void gst_mxnet_model_free(void)
{
    // Release NDList
    if (nd_hnd)
      MXNDListFree(nd_hnd);

    // Release Predictor
    MXPredFree(pred_hnd);
}

extern "C" int gst_mxnet_process_frame(
		void *gst_data,
		unsigned gst_width,
		unsigned gst_height)
{
	static int init_done = 0;
	if (!init_done) {
		init_done = 1;
		gst_mxnet_init_network();
	}

    // Read Image Data
    GetImageFile(gst_data, gst_width, gst_height, image_data.data(),
                 MODEL_CHANNELS, cv::Size(MODEL_WIDTH, MODEL_HEIGHT), nd_data);

    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), MODEL_IMAGE_SIZE);

    // Do Predict Forward
    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

	if (size > mxnet_out_data.size()) {
		std::cerr << "mxnet_out_data resizing to " << size << std::endl;
		mxnet_out_data.resize(size);
	}

    MXPredGetOutput(pred_hnd, output_index, &(mxnet_out_data[0]), size);

    // Print Output Data
    PrintOutputResult(mxnet_out_data, synset);

	if (0) {
		gst_mxnet_model_free();
	}

    return 0;
}
