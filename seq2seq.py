import logging
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel
import argparse
import torch

def load_conllu_dataset(datafile):
    arr = []
    with open(datafile, encoding='utf-8') as inp:
        strings = inp.readlines()
    for s in strings:
      if (s[0] != "#" and s.strip()):
          split_string = s.split('\t')
          arr.append([split_string[1] + " " + split_string[3]+ " " + split_string[5], split_string[2]])    
    return pd.DataFrame(arr, columns=["input_text", "target_text"])

def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

def main(args):
    train_df = load_conllu_dataset(args.train_data)
    eval_df = load_conllu_dataset(args.dev_data)
    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": max([len(token) for token in train_df["target_text"].tolist()]),
        "train_batch_size": int(args.batch),
        "num_train_epochs": int(args.epochs),
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        # "silent": True,
        "evaluate_generated_text": False,
        "evaluate_during_training": False,
        "evaluate_during_training_verbose": False,
        "use_multiprocessing": False,
        "save_best_model": False,
        "max_length": max([len(token) for token in train_df["input_text"].tolist()]),
        "save_steps": -1,
    }
    model = Seq2SeqModel(
        encoder_decoder_type=args.model_type,
        encoder_decoder_name=args.model, 
        args=model_args,
	use_cuda = torch.cuda.is_available(),)    
    model.train_model(train_df, eval_data=eval_df, matches=count_matches)
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data')
    parser.add_argument('--dev_data')
    parser.add_argument('--model_type', default="bart")
    parser.add_argument('--model', default="nbtpj/tiny-bart")
    parser.add_argument('--epochs', default="2")
    parser.add_argument('--batch', default="8")
    args = parser.parse_args()
    main(args)