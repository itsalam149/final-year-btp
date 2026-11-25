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
)

# Core JNI bridge library
add_library(
    llama-bridge
    SHARED
    llama-bridge.cpp
)

# Link against Android NDK core log and C++ STL libraries
find_library(log-lib log)

target_link_libraries(
    llama-bridge
    ${log-lib}
    atomic
)
