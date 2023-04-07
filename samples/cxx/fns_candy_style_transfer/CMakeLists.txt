# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


project(fns_candy_style_transfer)

include(../CMakeTemplate.txt)

if(LIBPNG_ROOTDIR)
  set(PNG_FOUND true)
  set(PNG_LIBRARIES png16)
  set(PNG_INCLUDE_DIRS "/usr/include")
  set(PNG_LIBDIR "${LIBPNG_ROOTDIR}")
else()
  message("Enabling png locating")
  find_package(PNG)
endif()


add_executable(${PROJECT_NAME}  "fns_candy_style_transfer.c" "image_file.h")
if(WIN32)
  message("Win32 ...")
  target_sources(${PROJECT_NAME}  PRIVATE image_file_wic.cc)
else()
  target_sources(${PROJECT_NAME}  PRIVATE image_file_libpng.c)
  message("cCC ...")
endif()

target_include_directories(${PROJECT_NAME}  PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}  PRIVATE onnxruntime ${PNG_LIBRARIES} ${WIL_LIB})
if(PNG_LIBDIR)
  target_link_directories(${PROJECT_NAME}  PRIVATE ${PNG_LIBDIR})
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
  set(CMAKE_VERBOSE_MAKEFILE 1)
  set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "Bits64_;UNIX;_BOOL;LINUX;FUNCPROTO;_GNU_SOURCE;LINUX_64;REQUIRE_IOSTREAM")
  set(GCC_COMPILE_OPTIONS "-m64;-fPIC;-fno-strict-aliasing;-Wall;-Wextra;-Wno-multichar;-Wno-comment;-Wno-sign-compare;-funsigned-char;-pthread;-Wno-deprecated;-Wno-reorder;-ftemplate-depth-64;-fno-gnu-keywords;-std=c++0x;-Winline")
  set(GCC_COMPILE_DEBUG_OPTIONS "${GCC_COMPILE_OPTIONS};-ggdb;-O0")
  target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${GCC_COMPILE_OPTIONS}>")
  target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Debug>:${GCC_COMPILE_OPTIONS}>")
  target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${GCC_COMPILE_OPTIONS}>")
  target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:${GCC_COMPILE_OPTIONS}>")
  target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:RelWithDebInfo>:${GCC_COMPILE_OPTIONS}>")
  target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:RelWithDebInfo>:${GCC_COMPILE_OPTIONS}>")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  set(MSVC_COMPILE_OPTIONS "/MP;/W3;/w34710;/Gy;/Zc:wchar_t;/nologo;/std:c++0x /EHsc")
  set(MSVC_COMPILE_DEBUG_OPTIONS "${MSVC_COMPILE_OPTIONS} /ZI /Od")
  set(MSVC_COMPILE_RELEASE_OPTIONS "${MSVC_COMPILE_OPTIONS} /O2")
  target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:${MSVC_COMPILE_DEBUG_OPTIONS}>")
  target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Release>:${MSVC_COMPILE_RELEASE_OPTIONS}>")
  message("cCC -O0 -g..............")
endif()
