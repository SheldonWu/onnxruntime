// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#define UNICODE
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>

#include "onnxruntime_cxx_api.h"
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


template <typename T>
static void softmax(T& input)
{
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
    MNIST(Ort::Session &session):session_(session)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                        input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                        output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run(const ORTCHAR_T* input_file)
    {
        int64_t ret{0};
        size_t input_height{0};
        size_t input_width{0};
        float  *inp_image_buf = NULL;
        size_t inp_buf_bytes {0};

        if (read_image_file(input_file, &input_height, &input_width, &inp_image_buf, &inp_buf_bytes) != 0) {
            return ret;
        }
        if (input_height != 28 || input_width != 28) {
            printf("please resize to image to 28x28\n");
            free(inp_image_buf);
            return ret;
        }
        //unit & float
        for (int i = 0; i<input_height*input_width; i++)
        {
            input_image_[i] = inp_image_buf[i]/255.0f;
        }

        const char* input_names[] = {"Input3"};
        const char* output_names[] = {"Plus214_Output_0"};

        Ort::RunOptions run_options;
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
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
    Ort::Session& session_;//{env, L"mnist.onnx", Ort::SessionOptions{nullptr}};

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 10};
};

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[])
#else
int main()
#endif
{
    std::unique_ptr<MNIST> mnist;
    try {
        Ort::Env env;
        Ort::Session session{env, (const ORTCHAR_T*)"/workspace/model/mnist/mnist.onnx", Ort::SessionOptions{nullptr}};
        mnist = std::make_unique<MNIST>(session);
        const char* image_file = "/workspace/data/mnist/test/9992-label-9.png";
        int64_t ret = mnist->Run(image_file);
        printf("Ret: %ld from %s\n", ret, image_file);
        for (unsigned i = 0; i < 10; i++) {
            float result = mnist->results_[i];
            printf("%2d: %d.%02d", i, int(result), abs(int(result * 100) % 100));
        }
        printf("\n");
    } catch (const Ort::Exception& exception) {
        printf("Error: %s", exception.what());
        return 0;
    }
}
