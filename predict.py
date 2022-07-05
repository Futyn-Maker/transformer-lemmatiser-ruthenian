import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel
import torch
import numpy as np
from itertools import cycle
import argparse
import os


def load_conllu_dataset(datafile):
    arr = []
    with open(datafile, encoding='utf-8') as inp:
        strings = inp.readlines()
    for s in strings:
      if (s[0] != "#" and s.strip()):
          split_string = s.split('\t')
          arr.append([split_string[1] + " " + split_string[3]+ " " + split_string[5], split_string[2]])    
    return pd.DataFrame(arr, columns=["input_text", "target_text"])


def main(args):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


    model = Seq2SeqModel(args.type, 'outputs/decoder', 'outputs/encoder', use_cuda = torch.cuda.is_available(),)
    
    eval_df = load_conllu_dataset(args.data)
    predictions = model.predict(eval_df["input_text"].tolist())
    predictions = cycle(predictions)
    with open(args.data, encoding='utf-8') as inp:
        strings = inp.readlines()
        predicted = []
        for s in strings:
          if (s[0] != "#" and s.strip()):
              split_string = s.split('\t')
              split_string[2] = next(predictions)
              joined_string = '\t'.join(split_string)
              predicted.append(joined_string)
              continue
          predicted.append(s)      
        with open(os.path.dirname(os.path.realpath(__file__)) + "/data/predictions_" + args.name + ".conllu", 'w', encoding='utf-8') as out:
          out.write(''.join(predicted))
          

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--type', default="bert")
    parser.add_argument('--model', default="facebook/bart-large")
    parser.add_argument('--name', default="my_data")
    args = parser.parse_args()
    main(args)