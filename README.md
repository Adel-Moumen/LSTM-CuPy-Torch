# LSTM-CuPy-Torch


run the LSTM on the adding toy task : python3 experiment/toy_task/add_task.py train --batch_size=8 --seq_length=200 --num_layers=2  


--------------------------------------------------------------
Model informations:
        Model: lstm
        Number of layers: 4
        Hidden size: 128
        seed: 42
        Training informations:
        Batch size: 8
        Learning rate: 0.001
        seq length: 200

LSTM PYTORCH 6.277
LSTM PYTORCH LOSS AVG 1.1832334995269775

LSTM JIT AUTOGRAD 122.476
LSTM JIT AUTOGRAD LOSS AVG 1.1814485788345337

LSTM AUTOGRAD 133.593
LSTM AUTOGRAD LOSS AVG 1.1814486980438232

LSTM CUSTOM 124.291
LSTM CUSTOM LOSS AVG 1.1814491748809814