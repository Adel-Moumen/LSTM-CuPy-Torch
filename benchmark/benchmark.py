
import torch
import torch.nn as nn

import config
import dataset
import engine


def run_benchmark(benchmark, model, name, seq_len):
    """
    :param model: torch.nn model 
    :param benchmark: BenchmarkTrain object
    :param name: string -> name of the object
    :param seq_len: seq_len 
    """

    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(hardware)
    print("hardware = ", hardware)
    model = model.to(device)
    engine.train(benchmark, model, device, name, seq_len)


if __name__ == "__main__":

    # set the seed 
    torch.manual_seed(config.SEED_NUMBER)

    # load models

    gru = nn.GRU(config.INPUT_SHAPE, config.HIDDEN_SIZE, config.NUM_LAYERS, batch_first=True)
    lstm = nn.LSTM(config.INPUT_SHAPE, config.HIDDEN_SIZE, config.NUM_LAYERS, batch_first=True)

    # iterate over seq_len to seq_len_max with a step of seq_len_step
    for seq_len in range(config.SEQ_LEN, config.SEQ_LEN_MAX, config.SEQ_LEN_STEP):

        # generate dataset
        X = torch.randn(1, config.BATCH_SIZE, seq_len,
                        config.INPUT_SHAPE)  # (numbers_input, seq_len, batch_size, input_shape)
        Y = torch.randn(1, config.BATCH_SIZE, seq_len, config.HIDDEN_SIZE)  # (seq_len, batch_size, input_shape)

        # create our BenchmarkTrain object
        benchmark_test = dataset.BenchmarkTrain(X, Y)

        torch.manual_seed(config.SEED_NUMBER)

        from src.lstm import LSTM
        lstm_vanilla = LSTM(
            input_shape=(config.BATCH_SIZE, seq_len, config.INPUT_SHAPE),
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
        )
        torch.manual_seed(config.SEED_NUMBER)

        from src.cupy_jit_autograd_lstm import LSTM
        lstm_cupy = LSTM(
            input_shape=(config.BATCH_SIZE, seq_len, config.INPUT_SHAPE),
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
        )

        lstm_jitted = torch.jit.script(lstm_vanilla)

        torch.manual_seed(config.SEED_NUMBER)
        lstm_pytorch  =  torch.nn.LSTM(
            input_size=config.INPUT_SHAPE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=0,
            bidirectional=False,
            bias=True,
            batch_first=True,
        )


        models = {"lstm_vanilla": lstm_vanilla, "lstm_cupy": lstm_cupy, "lstm_jitted": lstm_jitted, "lstm_pytorch": lstm_pytorch}

        for _, (name, model) in enumerate(models.items()):
            print("seq_len = {} ".format(seq_len))

            print("------------------------------------")
            run_benchmark(benchmark_test, model, name, seq_len)
            print("------------------------------------")