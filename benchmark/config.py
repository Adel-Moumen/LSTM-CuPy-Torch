
# train parameters
SEED_NUMBER = 42

# time step max that we will iterate over
SEQ_LEN_MAX = 6000

# numbers of step for time step
SEQ_LEN_STEP = 500

## Starting seq_len
SEQ_LEN = 100

VERSION = "v1"

# number of input
INPUT_NUMBER = 1

# input parameters
BATCH_SIZE = 1
INPUT_SHAPE = 1000# refractor to input_size ? 

# model parameters
NUM_LAYERS = 4
HIDDEN_SIZE = 1024

#CSV PATH
CSV_PATH = "/home/adel/Documents/ML/LSTM-CuPy-Torch/data/SeqLen" + str(SEQ_LEN_MAX) + "InputShape" + str(INPUT_SHAPE) + VERSION + ".csv"

#benchmark max number of iterations
ITERATOR = 10