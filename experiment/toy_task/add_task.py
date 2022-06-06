
import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, rnn, jit=False):
        super().__init__()

        self.rnn = rnn
        self.output_layer = nn.Linear(args.hidden_size, 1)
        self.jit = jit 
        if self.jit:
            self.rnn = torch.jit.script(self.rnn)

    def forward(self, input, hx=None):
        if hx is None:
            output, (h, c) = self.rnn(input)
        else:

            output, (h, c) = self.rnn(input, hx)

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



def train(net, delay_print, epochs):

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    mse_loss_fn = nn.MSELoss()

    net.train()
    minimum_loss = 0
    for epoch in range(epochs+1):

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
            minimum_loss += loss
            if epoch % delay_print == 0:
                print(f"loss = {loss}")
    
    return minimum_loss / args.epochs+1

if __name__ == "__main__":
    import time 
    Util.print_setup(args=args)

    delay_print = 100


    model_params = {
        'input_shape': (args.batch_size, args.seq_length, 2),
        'hiddden_size': args.hidden_size,
        'num_layers': args.num_layers,
    }  
    
    ########## CUSTOM 
    from src.cupy_jit_autograd_lstm import LSTM

    Util.seed_everything(args.seed)
    rnn_custom = LSTM(
        input_shape=model_params['input_shape'],
        hidden_size=model_params['hiddden_size'],
        num_layers=model_params['num_layers'],
    )
    delay_print = 100
    net = Model(rnn=rnn_custom, jit=False).to(args.device).float()

    #warmup
    loss_avg = train(net=net, delay_print=delay_print, epochs=1000)

    torch.cuda.synchronize()
    time1 = time.time()
    loss_avg = train(net=net, delay_print=delay_print, epochs=500)
    torch.cuda.synchronize()
    print(f"LSTM JIT AUTOGRAD+JIT+CUPY {(time.time() - time1):.3f}")
    print(f"LSTM JIT AUTOGRAD+JIT+CUPY LOSS AVG {loss_avg}")


    ########## CUSTOM 
    from src.lstm import LSTM

    Util.seed_everything(args.seed)
    rnn_custom = LSTM(
        input_shape=model_params['input_shape'],
        hidden_size=model_params['hiddden_size'],
        num_layers=model_params['num_layers'],
    )
    delay_print = 100
    net = Model(rnn=rnn_custom, jit=False).to(args.device).float()
    #warmup
    loss_avg = train(net=net, delay_print=delay_print, epochs=1000)

    torch.cuda.synchronize()
    time1 = time.time()
    loss_avg = train(net=net, delay_print=delay_print, epochs=500)
    torch.cuda.synchronize()
    print(f"LSTM CUSTOM (VANILLA) {(time.time() - time1):.3f}")
    print(f"LSTM CUSTOM (VANILLA) LOSS AVG {loss_avg}")



    ########## JIT + AUTOGRAD 
    from src.jit_autograd_lstm import LSTM

    Util.seed_everything(args.seed)

    rnn_custom = LSTM(
        input_shape=model_params['input_shape'],
        hidden_size=model_params['hiddden_size'],
        num_layers=model_params['num_layers'],
    )
    net = Model(rnn=rnn_custom, jit=False).to(args.device).float()

    #warmup
    loss_avg = train(net=net, delay_print=delay_print, epochs=10)
    torch.cuda.synchronize()
    time1 = time.time()
    loss_avg = train(net=net, delay_print=delay_print, epochs=500)
    torch.cuda.synchronize()
    print(f"LSTM JIT AUTOGRAD+JIT {(time.time() - time1):.3f}")
    print(f"LSTM JIT AUTOGRAD+JIT LOSS AVG {loss_avg}")

    ########## PYTORCH 
    Util.seed_everything(args.seed)
    rnn_pytorch =  torch.nn.LSTM(
        input_size=model_params['input_shape'][2],
        hidden_size=model_params['hiddden_size'],
        num_layers=model_params['num_layers'],
        dropout=0,
        bidirectional=False,
        bias=True,
        batch_first=True,
    )
    

    net = Model(rnn=rnn_pytorch).to(args.device).float()
    #warmup
    loss_avg = train(net=net, delay_print=delay_print, epochs=10)
    torch.cuda.synchronize()
    time1 = time.time()
    loss_avg = train(net=net, delay_print=delay_print, epochs=500)
    torch.cuda.synchronize()
    print(f"LSTM PYTORCH {(time.time() - time1):.3f}")
    print(f"LSTM PYTORCH LOSS AVG {loss_avg}")


    ########## AUTOGRAD 
    from src.autograd_lstm import LSTM

    Util.seed_everything(args.seed)

    rnn_custom = LSTM(
        input_shape=model_params['input_shape'],
        hidden_size=model_params['hiddden_size'],
        num_layers=model_params['num_layers'],
    )
    net = Model(rnn=rnn_custom, jit=False).to(args.device).float()
    #warmup
    loss_avg = train(net=net, delay_print=delay_print, epochs=10)
    torch.cuda.synchronize()
    time1 = time.time()
    loss_avg = train(net=net, delay_print=delay_print, epochs=500)
    torch.cuda.synchronize()
    print(f"LSTM AUTOGRAD (NO JIT) {(time.time() - time1):.3f}")
    print(f"LSTM AUTOGRAD (NO JIT) LOSS AVG {loss_avg}")

