// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#define UNICODE
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#ifdef _WIN32
#ifdef USE_DML
#include "providers.h"
#endif
#include <objbase.h>
#endif
#include "image_file.h"

#ifdef _WIN32
#define tcscmp wcscmp
#else
#define tcscmp strcmp
#endif

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
    do {                                                       \
        OrtStatus* onnx_status = (expr);                         \
        if (onnx_status != NULL) {                               \
        const char* msg = g_ort->GetErrorMessage(onnx_status); \
        fprintf(stderr, "%s\n", msg);                          \
        g_ort->ReleaseStatus(onnx_status);                     \
        abort();                                               \
        }                                                        \
    } while (0);

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

struct MNIST
{
    MNIST(OrtSession* session, OrtMemoryInfo* memory_info) : session_(session) {
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_image_.data(),
                                                                input_image_.size()*sizeof(float),
                                                                input_shape_.data(),
                                                                input_shape_.size(),
                                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                &input_tensor_));
        assert(input_tensor_ != NULL);
        int is_tensor;
        ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor_, &is_tensor));
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, results_.data(),
                                                                results_.size()*sizeof(float),
                                                                output_shape_.data(),
                                                                output_shape_.size(),
                                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                &output_tensor_));
        assert(output_tensor_ != NULL);
        ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor_, &is_tensor));
        assert(is_tensor);
    }

    std::ptrdiff_t Run(const ORTCHAR_T* input_file)  {
        int64_t ret{-1};
        if (g_ort == NULL)
            return ret;
        size_t input_height{0};
        size_t input_width{0};
        float* inp_image_buf = NULL;
        size_t inp_buf_bytes{0};

        if (read_image_file(input_file, &input_height, &input_width,
                &inp_image_buf, &inp_buf_bytes) != 0) {
            return ret;
        }
        if (input_height != 28 || input_width != 28) {
            printf("please resize to image to 28x28\n");
            free(inp_image_buf);
            return ret;
        }
        // unit & float
        for (int i = 0; i < input_height * input_width; i++) {
            input_image_[i] = inp_image_buf[i] / 255.0f;
        }

        const char* input_names[] = {"Input3"};
        const char* output_names[] = {"Plus214_Output_0"};
        int batch_num = 1;

        ORT_ABORT_ON_ERROR(g_ort->Run(session_, NULL, input_names,
                                    (const OrtValue* const*)&input_tensor_, batch_num,
                                    output_names, batch_num, &output_tensor_));

        // Ort::RunOptions run_options;
        // session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        softmax(results_);
        ret = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        free(inp_image_buf);
        return ret;
    }

    static constexpr const int width_ = 28;
    static constexpr const int height_ = 28;

    std::array<float, width_ * height_> input_image_{};
    std::array<float, 10> results_{};

private:
    OrtSession* session_;
    OrtValue* input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 1, width_, height_};
    OrtValue* output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 10};
};

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[])
#else
int main()
#endif
{
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    const char* model_path = "/workspace/model/mnist/mnist.onnx";
    const char* image_file = "/workspace/data/mnist/test/9998-label-5.png";

    OrtEnv* env{0};
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mnist", &env));
    assert(env != NULL);
    OrtSessionOptions* session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
    OrtSession* session{0};
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
    OrtMemoryInfo* memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    try {
        size_t count;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
        assert(count == 1);
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
        assert(count == 1);

        std::unique_ptr<MNIST> mnist = std::make_unique<MNIST>(session, memory_info);
        int64_t ret = mnist->Run(image_file);
        printf("Ret: %ld from %s\n", ret, image_file);
        for (unsigned i = 0; i < 10; i++) {
            float result = mnist->results_[i];
            printf("%2d: %d.%02d", i, int(result), abs(int(result * 100) % 100));
        }
        printf("\n");
    } catch (const Ort::Exception& exception) {
        printf("Error: %s", exception.what());
    }
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    return 0;
}
