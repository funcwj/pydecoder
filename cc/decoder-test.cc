#include <iostream>
#include "py-online-nnet3-decoder.h"


int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cerr << "command format error:" << std::endl
                  << "  usage: " << argv[0] << ": <decode-config> <nnet3-model> "
                  << "<decode-graph> <word-symbol> <input-wave>" << std::endl;
        return -1;
    }
    std::string decode_config(argv[1]), nnet3_model(argv[2]),
            decode_graph(argv[3]), word_symbol(argv[4]), input_wave(argv[5]);

    PyOnlineNnet3Decoder nnet3_decoder(decode_config, nnet3_model,
                                    decode_graph, word_symbol);

    WaveHolder holder;
    {
        Input ki(input_wave);
        holder.Read(ki.Stream());
    }

    const WaveData &wave_data = holder.Value();
    SubVector<BaseFloat> data(wave_data.Data(), 0);
    KALDI_LOG << data.Dim();
    nnet3_decoder.Decode(data.Data(), data.Dim(), wave_data.SampFreq());
    nnet3_decoder.Finalize();
    return 0;
}