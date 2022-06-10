# transformer-lemmatiser
Lemmatiser based on simpletransformers seq2seq model 

## Training

1. Clone the repository:
```
git clone https://github.com/The-One-Who-Speaks-and-Depicts/transformer-lemmatiser.git
```
2. Create the data directory within the repository directory
3. Add .conllu files for training and evaluation into the data directory.
4. Install Anaconda (recommended v11) and CUDA (recommended v11).
5. Follow simpletransformers installation instructions [here](https://github.com/ThilinaRajapakse/simpletransformers).
6. Activate virtual environment.
7. If you are going to use the local model (or have to work offline), save the model (for instance, clone [bart-large](https://huggingface.co/facebook/bart-large)) into the repository directory.
8. Run the model; for model_type and model reference, use [docs](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/seq2seq/seq2seq_model.py))
```
python3 seq2seq.py --train_data <path to the training data .conllu file> --dev_data <path to the evaluation data .conllu file> --model_type <string>  --model <model name, or path to the folder with model) --epochs <non-negative integer> --batch <non-negative integer>
```
9. Get the model from outputs directory after the run is finished.

## Evaluation/prediction

1. If you have a pre-trained model, put it into the Google Drive cloud.
2. Put the data files for evaluation and/or prediction into the Google Drive cloud.
3. Open in Google Colaboratory the file `Evaluation_prediction_script.ipynb`. Set GPU in the settings.
4. Run Package installation and loading and Supporting functions sections of the notebook.
5. Set `MODEL_NAME` variable to your pretrained model folder in Google Drive, or to huggingface model of choice. If your model is not bart, change `encoder_decoder_type` as well.
6. Run Model loading section.
7. For evaluation, set `EVAL_NAME` variable to path to the evaluation data (must be in .conllu format), and run the Model evaluation section. This produces text output, and .csv error report for accuracy score, Damerau-Levenshtein, Levenshtein and Jaro-Winkler distances metrics.
8. For evaluation, set `DATA_PRED_NAME` variable to path to the prediction data (must be in .conllu format), and run the Prediction section. This produces .conllu file with predicted lemmata.
