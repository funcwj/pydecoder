cmake_minimum_required(VERSION 3.5)
project(pydecoder)

set(CMAKE_CXX_STANDARD 11)

message("KALDI_ROOT = " $ENV{KALDI_ROOT})
message("FST_ROOT   = " $ENV{FST_ROOT})

set(TARGET online2-utt-nnet2-decoder)
set(SOURCE_FILES online2-utt-nnet2-decoder.cc)

# using OpenBLAS
include_directories($ENV{KALDI_ROOT}/src $ENV{KALDI_ROOT}/../OpenBLAS/install/include $ENV{FST_ROOT}/include)
link_directories($ENV{KALDI_ROOT}/src/lib $ENV{FST_ROOT}/lib)


add_definitions(-O3 -g -std=c++11 -DHAVE_OPENBLAS)
add_executable(${TARGET} ${SOURCE_FILES})
target_link_libraries(${TARGET} kaldi-base kaldi-util kaldi-matrix
                    kaldi-cudamatrix kaldi-nnet2 kaldi-ivector kaldi-decoder
                    kaldi-lat kaldi-fstext kaldi-hmm kaldi-feat kaldi-tree kaldi-gmm
                    kaldi-transform kaldi-online2 fst)
