Running RNN acceptor:
 - python experiment.py

Assumptions - bilstmTrain.py:
usage: bilstmTrain.py [-h] [--devFile DEV_PATH] [--vecFile VEC_PATH] [--vocabFile VOCAB_PATH] [--batchSize BATCH_SIZE] repr trainFile modelFile

LSTM Tagger

positional arguments:
  repr                  one of a,b,c,d
  trainFile             input file to train on
  modelFile             file to save the model

optional arguments:
  -h, --help            show this help message and exit
  --devFile DEV_PATH    dev file to calc acc during train
  --vecFile VEC_PATH    file to pretrained vectors
  --vocabFile VOCAB_PATH
                        file to the vocab of pretrained vectors
  --batchSize BATCH_SIZE
                        batch size to train. Default:5

* data files

