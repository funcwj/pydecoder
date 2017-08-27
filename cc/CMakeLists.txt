cmake_minimum_required(VERSION 3.4)
project(pydecoder)

set(CMAKE_CXX_STANDARD 11)

message("KALDI_ROOT = " $ENV{KALDI_ROOT})
message("FST_ROOT   = " $ENV{FST_ROOT})

set(TARGET decoder-test)
set(SOURCE_FILES decoder-test.cc py-online-nnet3-decoder.cc py-online-nnet3-decoder.h)

include_directories($ENV{KALDI_ROOT}/src $ENV{FST_ROOT}/include)
link_directories($ENV{KALDI_ROOT}/src/lib $ENV{FST_ROOT}/lib)

# using Accelerate framework in OSX
add_definitions(-O3 -g -std=c++11 -DHAVE_CLAPACK -framework Accelerate)
add_executable(${TARGET} ${SOURCE_FILES})
target_link_libraries(${TARGET} kaldi-base kaldi-util kaldi-matrix
        kaldi-cudamatrix kaldi-nnet3 kaldi-ivector kaldi-decoder
        kaldi-lat kaldi-fstext kaldi-hmm kaldi-feat kaldi-tree kaldi-gmm
        kaldi-transform kaldi-online2 fst)