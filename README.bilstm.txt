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

EXAMPLE:
python3 bilstmTrain.py c data/pos/train model_pos_c --devFile data/pos/dev --vecFile data/wordVectors.txt --vocabFile data/vocab.txt --batchSize 10

########################################################################################################################
########################################################################################################################
########################################################################################################################
usage: bilstmPredict.py [-h] repr modelFile inputFile

LSTM Tagger

positional arguments:
  repr        one of a,b,c,d
  modelFile   file to save the model
  inputFile   the blind input file to tag

optional arguments:
  -h, --help  show this help message and exit


EXAMPLE:
python3 bilstmPredict.py c model_ner data/ner/test

Note: The saving of test4.ner and test4.pos expect to receive the task-name within the path (e.g., ner/data/test)

