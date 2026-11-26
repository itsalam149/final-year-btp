cmake_minimum_required(VERSION 3.22.1)

project("llama-bridge")

# C++ Standard configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for ARMv8.2-A / NEON dot product
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    add_compile_options(
        -O3
        -DNDEBUG
        -march=armv8.2-a+dotprod+fp16
        -ffast-math
        -funroll-loops
    )
else()
    add_compile_options(
        -O3
        -DNDEBUG
    )
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Prebuilt libraries imported from jniLibs
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama.so)

add_library(llama-common SHARED IMPORTED)
set_target_properties(llama-common PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama-common.so)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml.so)

add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml-cpu.so)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml-base.so)

# Core JNI bridge library
add_library(
    llama-bridge
    SHARED
    llama-bridge.cpp
)

find_library(log-lib log)

target_link_libraries(
    llama-bridge
    llama
    llama-common
    ggml
    ggml-cpu
    ggml-base
    ${log-lib}
)
