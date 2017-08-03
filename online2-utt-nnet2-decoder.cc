#include "feat/wave-reader.h"
#include "online2/online-nnet2-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"



int main(int argc, const char *argv[]) {

    try {
        using namespace kaldi;
        using namespace fst;

        const char *usage =
                "Read in a single utterance in waveform and output the decoding sequence\n"
                "using nnet2 setup. We using the online logic in the course of decoding.\n"
                "Usage: online-decoder-utt-nnet2 [options] <nnet2-in> <fst-in> <wav-in> <word-syms-in>\n";

        ParseOptions po(usage);
        OnlineNnet2FeaturePipelineConfig feature_config;
        OnlineNnet2DecodingConfig decoding_config;

        feature_config.Register(&po);
        decoding_config.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            return 1;
        }

        std::string net_filename = po.GetArg(1),
                    fst_filename = po.GetArg(2),
                    utt_filename = po.GetArg(3),
                    word_sym_filename = po.GetArg(4);

        OnlineNnet2FeaturePipelineInfo feature_info(feature_config);

        TransitionModel trans_model;
        nnet2::AmNnet nnet;
        {
            bool binary;
            Input ki(net_filename, &binary);
            trans_model.Read(ki.Stream(), binary);
            nnet.Read(ki.Stream(), binary);
        }

        // defined in namespace fst
        // seems modified in kaldi-5.2
        fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_filename);
        fst::SymbolTable *word_syms = fst::SymbolTable::ReadText(word_sym_filename);

        if (!word_syms) {
            KALDI_ERR << "Could not read symbol table from file "
                      << word_sym_filename;
        }

        // A utterance for a speaker
        // OnlineIvectorExtractorAdaptationState adaptation_state(
        //      feature_info.ivector_extractor_info);

        WaveHolder holder;
        {
            Input ki(utt_filename);
            holder.Read(ki.Stream());
        }

        const WaveData &wave_data = holder.Value();
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        // feature_pipeline.SetAdaptationState(adaptation_state);

        SingleUtteranceNnet2Decoder decoder(decoding_config, trans_model,
                                            nnet, *decode_fst, &feature_pipeline);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_size = int32(samp_freq * 0.05);
        KALDI_ASSERT(chunk_size > 0);

        int32 cur_offset = 0;
        while (cur_offset < data.Dim()) {
            int32 remain_size = data.Dim() - cur_offset;
            int32 sample_num  = std::min(chunk_size, remain_size);
            SubVector<BaseFloat> batch(data, cur_offset, sample_num);
            feature_pipeline.AcceptWaveform(samp_freq, batch);
            cur_offset += sample_num;
            if (cur_offset == data.Dim())
                feature_pipeline.InputFinished();
            decoder.AdvanceDecoding();
        }

        decoder.FinalizeDecoding();
        CompactLattice clat;
        decoder.GetLattice(true, &clat);

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

        delete decode_fst;
        delete word_syms;
        return 1;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
