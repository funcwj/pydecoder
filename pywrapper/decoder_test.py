#!/usr/bin/env python

import argparse
import sys
import os
import pydecoder
import numpy as np 

def test_decoder():
    parser = argparse.ArgumentParser(description='test command for kaldi online decoder in nnet3 setup')
    parser.add_argument('config', type=str, help='config file for online decoding')
    parser.add_argument('model', type=str, help='location of nnet3 model(final.mdl)')
    parser.add_argument('graph', type=str, help='location of decoding graph(HCLG.fst)')
    parser.add_argument('words', type=str, help='dictionary for decoding sequence translation(words.txt)')
    parser.add_argument('wave', type=str, help='wave file for decoding')

    args = parser.parse_args()
    decoder = pydecoder.PyOnlineNnet3Decoder(args.config, args.model, args.graph)
    # decode whole file
    decoder.decode_wavefile(args.wave)
    # and get results
    print decoder.get_decode_sequence(args.words)

if __name__ == '__main__':
    test_decoder()