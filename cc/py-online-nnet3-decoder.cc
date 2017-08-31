//
// Created by wujian on 14/08/2017.
//

#include "py-online-nnet3-decoder.h"


PyOnlineNnet3Decoder::PyOnlineNnet3Decoder(std::string config,
                                           std::string nnet3_path,
                                           std::string fst_path,
                                           std::string wordsym_path) {
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
    word_syms_  = fst::SymbolTable::ReadText(wordsym_path);

    feature_info_ = new OnlineNnet2FeaturePipelineInfo(feature_config);
    decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts, &am_nnet_);
    feature_pipeline_ = new OnlineNnet2FeaturePipeline(*feature_info_);
    decodable_ = new nnet3::DecodableAmNnetLoopedOnline(trans_model_, *decodable_info_,
                                                        feature_pipeline_->InputFeature(),
                                                        feature_pipeline_->IvectorFeature());
    decoder_ = new LatticeFasterOnlineDecoder(decoder_config, decode_fst_);
    decoder_->InitDecoding();

}

PyOnlineNnet3Decoder::~PyOnlineNnet3Decoder() {
    delete feature_info_;
    delete word_syms_;
    delete decodable_info_;
    delete feature_pipeline_;
    delete decoder_;    // delete decode_fst also
}


void PyOnlineNnet3Decoder::Decode(BaseFloat *waveform, int32 sample_size, BaseFloat sampling_rate) {
    SubVector<BaseFloat> wave_part(waveform, sample_size);
    feature_pipeline_->AcceptWaveform(sampling_rate, wave_part);
    decoder_->AdvanceDecoding(decodable_);
}

void PyOnlineNnet3Decoder::Finalize() {

    decoder_->FinalizeDecoding();

    Lattice raw_lat;
    decoder_->GetRawLattice(&raw_lat, true);

    KALDI_LOG << "lattice-beam: " << decoder_config.lattice_beam;
    KALDI_LOG << "min-active: " << decoder_config.min_active;
    KALDI_LOG << "max-active: " << decoder_config.max_active;

    CompactLattice clat;
    DeterminizeLatticePhonePrunedWrapper(
            trans_model_, &raw_lat, decoder_config.lattice_beam, &clat, decoder_config.det_opts);

    while (true) {

        if (clat.NumStates() == 0) {
            KALDI_LOG << "Got empty lattice.";
            break;
        }
        CompactLattice best_path_clat;
        CompactLatticeShortestPath(clat, &best_path_clat);

        Lattice best_path_lat;
        ConvertLattice(best_path_clat, &best_path_lat);

        std::vector<int32> words;
        std::vector<int32> alignment;
        LatticeWeight weight;
        double likelihood;

        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
        likelihood = -(weight.Value1() + weight.Value2());

        for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms_->Find(words[i]);
            KALDI_ASSERT(s != "");
            std::cerr << s << ' ';
        }
        std::cerr << "[Average likelihood = " << likelihood / alignment.size() << "]" << std::endl;
        break;
    }
}