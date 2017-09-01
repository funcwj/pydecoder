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


class OnlineNnet3Decoder {
public:
    OnlineNnet3Decoder(std::string config, std::string nnet3,
                        std::string fst);
    ~OnlineNnet3Decoder();
    void DecodeWaveform(BaseFloat *waveform, int32 sample_size, BaseFloat sampling_rate);
    void GetDecodeSequence(std::vector<int32> *words);

private:
    // config setup

    // struct, could be local
    OnlineNnet2FeaturePipelineConfig feature_config;

    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_config;


    // argument setup
    TransitionModel trans_model_;
    nnet3::AmNnetSimple am_nnet_;
    fst::Fst<fst::StdArc> *decode_fst_;
    // fst::SymbolTable *word_syms_;

    // decode setup
    // using param:
    //   const OnlineNnet2FeaturePipelineConfig &;
    OnlineNnet2FeaturePipelineInfo *feature_info_;
    // using param:
    //   const OnlineNnet2FeaturePipelineInfo &;
    OnlineNnet2FeaturePipeline *feature_pipeline_;

    // using param:
    //    const NnetSimpleLoopedComputationOptions &,
    //    AmNnetSimple *
    nnet3::DecodableNnetSimpleLoopedInfo *decodable_info_;

    // using param:
    //   const DecodableNnetSimpleLoopedInfo &,
    //   OnlineFeatureInterface *input_features,
    //   OnlineFeatureInterface *ivector_features
    nnet3::DecodableAmNnetLoopedOnline *decodable_;
    // using param:
    //  const LatticeFasterDecoderConfig &;
    //  fst::Fst<fst::StdArc> * (delete the fst when decoder_ destroyed);
    LatticeFasterOnlineDecoder *decoder_;
};


#endif // PYONLINENNET3DECODER_H
