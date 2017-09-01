import cython
import wave
import numpy as np
import pywrapfst

from cython cimport address
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport numpy as np
from libc.stdint cimport *


cdef extern from "<fst/types.h>":
    ctypedef int32_t int32

cdef extern from '../cc/py-online-nnet3-decoder.h':
    cdef cppclass OnlineNnet3Decoder:
        OnlineNnet3Decoder(string, string, string)
        void DecodeWaveform(float*, int32, float)
        void GetDecodeSequence(vector[int32]*)


cdef class PyOnlineNnet3Decoder:
    cdef OnlineNnet3Decoder *online_decoder

    def __cinit__(self, string config, string nnet3, string fst):
        self.online_decoder = new OnlineNnet3Decoder(config, nnet3, fst)

    def __dealloc__(self):
        del self.online_decoder
    
    def get_decode_sequence(self, words):
        cdef vector[int32] words_id
        self.online_decoder.GetDecodeSequence(address(words_id))
        words_symbols = pywrapfst.SymbolTable.read_text(words)
        sequence = ''
        for idx in range(words_id.size()):
            sequence += words_symbols.find(words_id[idx])
        return sequence
        
    def decode_waveform(self, np.ndarray[float, ndim=1] waveform, float sampling_rate):
        self.online_decoder.DecodeWaveform(<float*>waveform.data, waveform.size, sampling_rate)

    def decode_wavefile(self, string wave_file):
        wf = wave.open(wave_file, 'rb')
        sampling_rate = wf.getframerate()
        sampling_size = wf.getnframes()
        # from int16 to float32
        waveform = np.array(np.fromstring(wf.readframes(sampling_size), dtype=np.int16), 
                            dtype=np.float32)
        self.decode_waveform(waveform, sampling_rate)
