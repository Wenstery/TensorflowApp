/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_jni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "jni_utils.h"
#include "tensorflow/c/c_api_internal.h"

using namespace tensorflow;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Global variables that holds the TensorFlow classifier.
#define MAX_NUM 10
static std::unique_ptr <tensorflow::Session> session[MAX_NUM];
static tensorflow::GraphDef tensorflow_graph[MAX_NUM];
static string input_layer = "image_tensor:0";
static string d_boxes = "detection_boxes:0";
static string d_scores = "detection_scores:0";
static string d_num = "num_detections:0";

static bool g_compute_graph_initialized[MAX_NUM]={false};
// static mutex g_compute_graph_mutex(base::LINKER_INITIALIZED);

jmethodID addFacesCallbackId;


inline static int64 CurrentThreadTimeUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(initializeTensorFlow)(
        JNIEnv *env, jobject thiz, jobject java_asset_manager, jstring model, jint threadId) {
    // MutexLock input_lock(&g_compute_graph_mutex);
    if (g_compute_graph_initialized[threadId]) {
        LOG(INFO) << "Compute graph already loaded. skipping.";
        return 0;
    }

    const char *const model_cstr = env->GetStringUTFChars(model, NULL);

    LOG(INFO) << "Loading TensorFlow.";

    LOG(INFO) << "Making new SessionOptions.";
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto &config = options.config;
    google::protobuf::Map<std::string, google::protobuf::int32>::const_iterator it;
    config.mutable_device_count()->insert(google::protobuf::MapPair<std::string, google::protobuf::int32>("cpu",8));
    config.set_inter_op_parallelism_threads (1);
    config.set_intra_op_parallelism_threads(8);
    for (it = config.device_count().begin(); it != config.device_count().end(); it++) {
        LOG(INFO) << "Got config,core " << it->second << " devices";
    }
    for (it = config.mutable_device_count()->begin(); it != config.mutable_device_count()->end(); it++) {
        LOG(INFO) << "Got config,core mutable " << it->second << " devices";
    }
    LOG(INFO) << "Got config, " << config.device_count_size() << " devices config";
    LOG(INFO) << "Got config, " << config.inter_op_parallelism_threads() << " inter threads";
    LOG(INFO) << "Got config, " << config.intra_op_parallelism_threads() << " intra threads";

    session[threadId].reset(tensorflow::NewSession(options));
    LOG(INFO) << "Session created.";

    AAssetManager *const asset_manager =
            AAssetManager_fromJava(env, java_asset_manager);
    LOG(INFO) << "Acquired AssetManager.";

    LOG(INFO) << "Reading file to proto: " << model_cstr;
    ReadFileToProto(asset_manager, model_cstr, &tensorflow_graph[threadId]);

    LOG(INFO) << "Creating session.";
    tensorflow::Status s = session[threadId]->Create(tensorflow_graph[threadId]);
    if (!s.ok()) {
        LOG(INFO) << "Could not create TensorFlow Graph: " << s;
    }

    // Clear the proto to save memory space.
    tensorflow_graph[threadId].Clear();
    LOG(INFO) << "TensorFlow graph loaded from: " << model_cstr;

    g_compute_graph_initialized[threadId] = true;

    LOG(INFO) << "Initialization done in ";

    //jclass detectorClass = env->FindClass("com/via/newwebcam_facedetect/SsdDetector");

    //addFacesCallbackId = env->GetMethodID(detectorClass, "addFacesCallback", "(FFFFF)V");

    return 0;
}

void Deallocator(void* data, size_t size, void* arg) {
    tensorflow::cpu_allocator()->DeallocateRaw(data);
    *reinterpret_cast<bool*>(arg) = true;
}

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(detectFrame)(
        JNIEnv *env, jobject thiz, jintArray pixels, jint width, jint height, jfloat threshold, jint threadId) {
    jboolean iCopied = JNI_FALSE;
    jint *frameData = env->GetIntArrayElements(pixels, &iCopied);
    uint8_t *u_frame = new uint8_t[width*height*3];
    for(int m = 0; m< width*height*3; m++){
        u_frame[m] = static_cast<uint8_t>(frameData[m]);
    }
    /*
    // Create input tensor
    tensorflow::Tensor input_tensor(
            tensorflow::DT_UINT8,
            tensorflow::TensorShape(
                    {1, width, height, 3}));
    auto input_tensor_mapped = input_tensor.tensor<uint8_t, 4>();
    //LOG(INFO) << "TensorFlow: Copying Data.";
    int64 startTime = CurrentThreadTimeUs();
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            input_tensor_mapped(0, i, j, 0) =
                    static_cast<uint8_t>(frameData[(i * width + j) * 3 + 0]);
            input_tensor_mapped(0, i, j, 1) =
                    static_cast<uint8_t >(frameData[(i * width + j) * 3 + 1]);
            input_tensor_mapped(0, i, j, 2) =
                    static_cast<uint8_t >(frameData[(i * width + j) * 3 + 2]);
        }
    }
    int64 endTime = CurrentThreadTimeUs();
    LOG(ERROR) << "Running copy data to tensor cost : " << endTime - startTime<<"ms";

    std::vector <std::pair<std::string, tensorflow::Tensor>> input_tensors(
            {{input_layer, input_tensor}});
    */

    //VLOG(0) << "Start computing.";
    int64 tensorDims[4] = {1, width, height, 3};
    bool deallocator_called = false;
    TF_Tensor* tfTensor = TF_NewTensor(TF_DataType::TF_UINT8, tensorDims, 4,
                                       u_frame, width * height * 3 ,
                                       &Deallocator, &deallocator_called);
    Tensor input_tensor = tensorflow::TensorCApi::MakeTensor(tfTensor->dtype, tfTensor->shape, tfTensor->buffer);
    std::vector <std::pair<std::string, tensorflow::Tensor>> input_tensors(
            {{input_layer, input_tensor}});
    std::vector <Tensor> outputs;
    int64 start_time = CurrentThreadTimeUs();
    Status run_status = session[threadId]->Run(input_tensors,
                                     {d_boxes, d_scores, d_num}, {}, &outputs);
    int64 end_time = CurrentThreadTimeUs();
    LOG(ERROR) << "Running model cost : " << (end_time - start_time)/1000000.0<<"s";
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    //tensorflow::TTypes<float>::Flat boxes_flat = outputs[0].flat_outer_dims<float, 1>();
    tensorflow::TTypes<float>::Flat score_flat = outputs[1].flat<float>();
    //LOG(ERROR) << "Running model result: " << "ymin1:" << ymin1 << ",xmin1:" << xmin1 << ",ymax1:"
    //          << ymax1 << ",xmax1:" << xmax1;
    float score = score_flat(0);
    LOG(ERROR) << "Running model result: " << "score0:" << score;
    /*
    for (int i = 0; i < 10; i++) {
        float score = score_flat(i);
        //if (score < threshold) {
        //    break;
        //}
        float ymin, xmin, ymax, xmax = 0;
        ymin = boxes_flat(i);
        xmin = boxes_flat(i + 1);
        ymax = boxes_flat(i + 2);
        xmax = boxes_flat(i + 3);

        //env->CallVoidMethod(thiz, addFacesCallbackId, ymin*height, xmin*width, ymax*height, xmax*width, score);

    }*/
    env->ReleaseIntArrayElements(pixels, frameData, JNI_ABORT);
    return 0;

}




