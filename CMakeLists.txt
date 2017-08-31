cmake_minimum_required(VERSION 3.5)
project(pydecoder)

set(CMAKE_CXX_STANDARD 11)

set(KALDI_ROOT /home/wujian/Document/git/kaldi-5.2)
set(FST_ROOT   /home/wujian/Document/git/kaldi-5.2/tools/openfst)
# message("KALDI_ROOT = " $ENV{KALDI_ROOT})
# message("FST_ROOT   = " $ENV{FST_ROOT})

set(TARGET online2-utt-nnet3-decoder)
set(SOURCE_FILES online2-utt-nnet3-decoder.cc)

include_directories(${KALDI_ROOT}/src ${KALDI_ROOT}/../OpenBLAS/install/include ${FST_ROOT}/include)
link_directories(${KALDI_ROOT}/src/lib ${FST_ROOT}/lib)

# -Wno-deprecated-declarations: some deprecated symbols in fst
add_definitions(-O3 -g -std=c++11 -DHAVE_OPENBLAS -Wno-deprecated-declarations)
add_executable(${TARGET} ${SOURCE_FILES})
target_link_libraries(${TARGET} kaldi-base kaldi-util kaldi-matrix
        kaldi-cudamatrix kaldi-nnet3 kaldi-ivector kaldi-decoder
        kaldi-lat kaldi-fstext kaldi-hmm kaldi-feat kaldi-tree kaldi-gmm
        kaldi-transform kaldi-online2 fst)
