cmake_minimum_required(VERSION 3.15)
project(streaming-piper LANGUAGES CXX)

option(USE_RKNN "Enable RKNN for accelerated inference" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(ORT_ROOT "" CACHE STRING "Path to ONNX Runtime root directory")
if (ORT_ROOT STREQUAL "")
    message(FATAL_ERROR "ORT_ROOT must be set")
endif()

include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

set(PIPER_PHONEMIZE_ROOT "" CACHE STRING "Path to eSpeak NG root directory")
if (PIPER_PHONEMIZE_ROOT STREQUAL "")
    message(FATAL_ERROR "PIPER_PHONEMIZE_ROOT must be set")
endif()

include_directories(${PIPER_PHONEMIZE_ROOT}/include)
link_directories(${PIPER_PHONEMIZE_ROOT}/lib)

find_package(Drogon REQUIRED)
find_package(fmt REQUIRED)
find_package(Opus REQUIRED)

add_library(pipercore
    pipercore/piper.cpp)

if (USE_RKNN)
    target_compile_definitions(pipercore PRIVATE USE_RKNN)
    target_sources(pipercore PRIVATE pipercore/rknn-inferer.cpp)
endif()

target_precompile_headers(pipercore PRIVATE pipercore/pch.hpp)

target_link_libraries(pipercore
    PRIVATE
    onnxruntime
    espeak-ng
    piper_phonemize
    pthread
    fmt::fmt
    dl
    rknnrt)

target_include_directories(pipercore
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/pipercore)

add_executable(piper-cli
    piper-cli/main.cpp)
target_link_libraries(piper-cli pipercore)

add_executable(piper-server
    piper-server/api.cpp
    piper-server/OggOpusEncoder.cpp
    piper-server/main.cpp)
target_link_libraries(piper-server PRIVATE pipercore Drogon::Drogon soxr ${OPUS_LIBRARIES} opusenc ogg)
target_include_directories(piper-server PRIVATE ${OPUS_INCLUDE_DIRS})
target_precompile_headers(piper-server PRIVATE piper-server/pch.hpp)

