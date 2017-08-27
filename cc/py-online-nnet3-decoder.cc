//
// Created by wujian on 14/08/2017.
//

#include "py-online-nnet3-decoder.h"


PyOnlineNnet3Decoder::PyOnlineNnet3Decoder(std::string config,
                                           std::string nnet3_path,
                                           std::string fst_path,
                                           std::string wordsym_path) {
    ConfigSetup(config);
    ResourceSetup(nnet3_path, fst_path, wordsym_path);
    DecodeSetup();
}

PyOnlineNnet3Decoder::~PyOnlineNnet3Decoder() {
    delete decode_fst;
    delete word_syms;
    delete decodable_info;
    delete feature_pipeline;
    delete decoder;
}

void PyOnlineNnet3Decoder::ConfigSetup(std::string config) {
    ParseOptions po("");
    feature_config.Register(&po);
    decodable_opts.Register(&po);
    decoder_config.Register(&po);
    po.ReadConfigFile(config);
}

void PyOnlineNnet3Decoder::ResourceSetup(std::string nnet3_path,
                                         std::string fst_path,
                                         std::string wordsym_path) {
    bool binary;
    Input ki(nnet3_path, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
    SetDropoutTestMode(true, &(am_nnet.GetNnet()));
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));

    decode_fst = ReadFstKaldiGeneric(fst_path);
    word_syms = fst::SymbolTable::ReadText(wordsym_path);
}


void PyOnlineNnet3Decoder::DecodeSetup() {
    OnlineNnet2FeaturePipelineInfo feature_info(feature_config);
    decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts, &am_nnet);
    feature_pipeline = new OnlineNnet2FeaturePipeline(feature_info);
    decoder = new SingleUtteranceNnet3Decoder(decoder_config, trans_model,
            *decodable_info, *decode_fst, feature_pipeline);
}

void PyOnlineNnet3Decoder::Decode(float *waveform, int32 sample_size) {
    SubVector<BaseFloat> wave_part(waveform, sample_size);
    feature_pipeline->AcceptWaveform(16000, wave_part);
    decoder->AdvanceDecoding();
}

void PyOnlineNnet3Decoder::Finalize() {

    decoder->FinalizeDecoding();
    CompactLattice clat;
    decoder->GetLattice(true, &clat);
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
            std::string s = word_syms->Find(words[i]);
            KALDI_ASSERT(s != "");
            std::cerr << s << ' ';
        }
        std::cerr << "[Average likelihood = " << likelihood / alignment.size() << "]" << std::endl;
        break;
    }
}