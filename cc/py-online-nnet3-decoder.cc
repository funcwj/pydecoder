//
// Created by wujian on 14/08/2017.
//

#include "py-online-nnet3-decoder.h"


OnlineNnet3Decoder::OnlineNnet3Decoder(std::string config,
                                           std::string nnet3_path,
                                           std::string fst_path) {
    ParseOptions po("");
    feature_config.Register(&po);
    decodable_opts.Register(&po);
    decoder_config.Register(&po);
    po.ReadConfigFile(config);

    bool binary;
    Input ki(nnet3_path, &binary);
    trans_model_.Read(ki.Stream(), binary);
    am_nnet_.Read(ki.Stream(), binary);

    nnet3::Nnet *nnet = &(am_nnet_.GetNnet());
    SetBatchnormTestMode(true, nnet);
    SetDropoutTestMode(true, nnet);
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), nnet);

    decode_fst_ = ReadFstKaldiGeneric(fst_path);

    feature_info_ = new OnlineNnet2FeaturePipelineInfo(feature_config);
    decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts, &am_nnet_);
    feature_pipeline_ = new OnlineNnet2FeaturePipeline(*feature_info_);
    decodable_ = new nnet3::DecodableAmNnetLoopedOnline(trans_model_, *decodable_info_,
                                                        feature_pipeline_->InputFeature(),
                                                        feature_pipeline_->IvectorFeature());
    decoder_ = new LatticeFasterOnlineDecoder(decoder_config, decode_fst_);
    decoder_->InitDecoding();

}

OnlineNnet3Decoder::~OnlineNnet3Decoder() {
    delete feature_info_;
    delete decodable_info_;
    delete feature_pipeline_;
    delete decoder_;    // delete decode_fst also
}


void OnlineNnet3Decoder::DecodeWaveform(BaseFloat *waveform,
                                  int32 sample_size,
                                  BaseFloat sampling_rate) {

    SubVector<BaseFloat> wave_part(waveform, sample_size);
    feature_pipeline_->AcceptWaveform(sampling_rate, wave_part);
    feature_pipeline_->InputFinished();
    decoder_->AdvanceDecoding(decodable_);
}

void OnlineNnet3Decoder::GetDecodeSequence(std::vector<int32> *words) {

    decoder_->FinalizeDecoding();

    Lattice raw_lat;
    decoder_->GetRawLattice(&raw_lat, true);
    CompactLattice clat;
    DeterminizeLatticePhonePrunedWrapper(
            trans_model_, &raw_lat, decoder_config.lattice_beam,
            &clat, decoder_config.det_opts);

    KALDI_ASSERT(clat.NumStates());

    CompactLattice best_path_clat;
    CompactLatticeShortestPath(clat, &best_path_clat);

    Lattice best_path_lat;
    ConvertLattice(best_path_clat, &best_path_lat);

    std::vector<int32> alignment;
    LatticeWeight weight;
    GetLinearSymbolSequence(best_path_lat, &alignment, words, &weight);

    KALDI_LOG << "[Average likelihood = "
              << -(weight.Value1() + weight.Value2()) / alignment.size()
              << "]" << std::endl;
}