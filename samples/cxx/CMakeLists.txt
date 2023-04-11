# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(ort_samples)


set(OSS_SAMPLES_LIST
    fns_candy_style_transfer
    mnist
    )

foreach(SAMPLE_ITER ${OSS_SAMPLES_LIST})
    message("code: " ${SAMPLE_ITER})
    add_subdirectory(${SAMPLE_ITER})
endforeach(SAMPLE_ITER)
