cmake_minimum_required(VERSION 3.4)
project(pydecoder)

set(CMAKE_CXX_STANDARD 11)

# could using value from env
message("KALDI_ROOT = " $ENV{KALDI_ROOT})
message("FST_ROOT   = " $ENV{FST_ROOT})

set(KALDI_ROOT /home/wujian/Document/git/kaldi-5.2)
set(FST_ROOT ${KALDI_ROOT}/tools/openfst)

set(TARGET pydecoder3)
set(SOURCE_FILES py-online-nnet3-decoder.cc py-online-nnet3-decoder.h)


# in Ubuntu using OpenBLAS
include_directories(${KALDI_ROOT}/src ${KALDI_ROOT}/../OpenBLAS/install/include ${FST_ROOT}/include)
link_directories(${KALDI_ROOT}/src/lib ${FST_ROOT}/lib)

# in OSX using Accelerate framework
# include_directories($ENV{KALDI_ROOT}/src $ENV{FST_ROOT}/include)
# add_definitions(-O3 -g -std=c++11 -DHAVE_CLAPACK -framework Accelerate)

add_definitions(-O3 -g -std=c++11 -DHAVE_OPENBLAS -Wno-deprecated-declarations)
# add_executable(${TARGET} ${SOURCE_FILES})
add_library(${TARGET} SHARED ${SOURCE_FILES})
# set_target_properties(${TARGET} PROPERTIES PREFIX "")
target_link_libraries(${TARGET} kaldi-base kaldi-util kaldi-matrix
        kaldi-cudamatrix kaldi-nnet3 kaldi-ivector kaldi-decoder
        kaldi-lat kaldi-fstext kaldi-hmm kaldi-feat kaldi-tree kaldi-gmm
        kaldi-transform kaldi-online2 fst)
