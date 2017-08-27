//
// Created by wujian on 14/08/2017.
//

#ifndef PYONLINENNET3DECODER_H
#define PYONLINENNET3DECODER_H

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"

using namespace kaldi;
using namespace fst;


class PyOnlineNnet3Decoder {
public:
    PyOnlineNnet3Decoder(std::string config, std::string nnet3,
                        std::string fst, std::string wordsym);
    ~PyOnlineNnet3Decoder();
    void ConfigSetup(std::string config);
    void ResourceSetup(std::string nnet3, std::string fst,
                       std::string wordsym);
    void DecodeSetup();
    void Decode(float *waveform, int32 sample_size);
    void Finalize();

private:
    // config setup
    OnlineNnet2FeaturePipelineConfig feature_config;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_config;

    // argument setup
    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    fst::Fst<fst::StdArc> *decode_fst;
    fst::SymbolTable *word_syms;

    // decode setup
    SingleUtteranceNnet3Decoder *decoder;
    OnlineNnet2FeaturePipeline *feature_pipeline;
    nnet3::DecodableNnetSimpleLoopedInfo *decodable_info;

};


#endif // PYONLINENNET3DECODER_H
