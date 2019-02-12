#@opensource

import configparser
import glob
import os
import sentencepiece as sp

CONFIGPATH ='config.ini'
config = configparser.ConfigParser()
config.read(CONFIGPATH)

FILE = config['DATA']['TEXTDIR']
PREFIX = config['SENTENCEPIECE']['PREFIX']
VOCABSIZE = config['SENTENCEPIECE']['VOCABSIZE']
CTLSYMBOLS = config['SENTENCEPIECE']['CTLSYMBOLS']


def train(prefix=PREFIX, vocab_size=VOCABSIZE, ctl_symbols=CTLSYMBOLS,file=FILE):
    command = f'--input={file} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols}'
    sp.SentencePieceTrainer.Train(command)


def main():
    try:
       os.mkdir('sentence-piece_output')
    except:
       pass
    train()


if __name__ == "__main__":
    main()
