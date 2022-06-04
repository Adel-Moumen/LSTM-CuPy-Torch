
import torch
import torch.nn as nn
import torch.optim as optim

from src.lstm import LSTM
from util import Util

import numpy as np
import random
from util import Util
import argparse



main_arg_parser = argparse.ArgumentParser(description="parser for training differents LiGRU on adding task")
subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

model_arg_parser = subparsers.add_parser("train", help="parser for model")
model_arg_parser.add_argument("--model", type=str, default="lstm",
help="model, default is lstm")
model_arg_parser.add_argument("--hidden_size", type=int, default=128,
help="hidden size, default is 128")
model_arg_parser.add_argument("--num_layers", type=int, default=1,
help="num layers, default is 1")
model_arg_parser.add_argument("--learning_rate", type=float, default=0.001,
help="learning rate, default is 0.001")
model_arg_parser.add_argument("--epochs", type=int, default=2000,
                                help="number of training epochs, default is 2000")
model_arg_parser.add_argument("--device", type=str, default="cuda",
                                help="device, default is cuda")
model_arg_parser.add_argument("--batch_size", type=int, default=256,
                                help="batch size for training, default is 256")
model_arg_parser.add_argument("--seed", type=int, default=42,
                                help="seed for training, default is 42")
model_arg_parser.add_argument("--seq_length", type=int, default=200,
                                help="sequence length for training, default is 200")
  

args = main_arg_parser.parse_args()

class Model(nn.Module):
    def __init__(self, rnn):
        super().__init__()
        self.rnn = torch.jit.script(rnn)
        self.output_layer = nn.Linear(args.hidden_size, 1)
        self.ht = None

    def forward(self, input, hx=None):
        if hx is None:
            output, h, c = self.rnn(input)
        else:

            output,  h, c = self.rnn(input, hx)

        output = self.output_layer(output[:, -1, :])
        return output

def generate_add_example(seq_length):
    b1 = random.randint(0, seq_length//2 - 1)
    b2 = random.randint(seq_length//2, seq_length - 1)
    
    mask = [0.] * seq_length
    mask[b1] = 1.
    mask[b2] = 1.

    x = [(random.uniform(0, 1), marker) for marker in mask]
    y = x[b1][0] + x[b2][0]
    
    return x, y

def generate_batch(seq_length, batch_size):
    
    n_elems = 2
    x = np.empty((batch_size, seq_length, n_elems))
    y = np.empty((batch_size, 1))

    for i in range(batch_size):
        sample, ground_truth = generate_add_example(seq_length=seq_length)
        x[i, :, :] = sample 
        y[i, 0] = ground_truth
    return x, y


if __name__ == "__main__":
    Util.print_setup(args=args)
    Util.seed_everything(args.seed)

    model_params = {
        'input_shape': (args.batch_size, args.seq_length, 2),
        'hiddden_size': args.hidden_size,
        'num_layers': args.num_layers,
    }  
    

    rnn = LSTM(
        input_shape=model_params['input_shape'],
        hidden_size=model_params['hiddden_size'],
        num_layers=model_params['num_layers'],
    )

    net = Model(rnn=rnn).to(args.device).float()

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    mse_loss_fn = nn.MSELoss()

    delay_print = 50

    net.train()
    for epoch in range(args.epochs+1):

        x, y = generate_batch(
            seq_length=args.seq_length,
            batch_size=args.batch_size,
        )

        x = torch.tensor(x, device=args.device, requires_grad=True).float()
        y = torch.tensor(y, device=args.device, requires_grad=False).float()
        
        out = net(x)

        loss = mse_loss_fn(out, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
        
            if epoch % delay_print == 0:
                print(f"loss = {loss}")
                