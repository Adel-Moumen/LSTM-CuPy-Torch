import torch
import torch.nn as nn 
import torch.autograd as autograd 
import torch.nn.functional as F

from typing import Optional, Tuple


class LSTM_Layer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_size: int,
        dropout=0.0,
        bias=True,
        bidirectional=False,
    ):
        super(LSTM_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bias = bias
        self.num_layers = num_layers

        self.w = nn.Linear(self.input_size, 4 * self.hidden_size, bias=self.bias)
        self.u = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=self.bias)

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size))

        # Preloading dropout masks (gives some speed improvement)
        # TODO add init_drop
        #self._init_drop(self.batch_size)

    def _lstm_cell(self, x: torch.Tensor, ht: torch.Tensor, ct: torch.Tensor):
        """Returns the hidden states for each time step.
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        hiddens = []
        cell_state = []

        # Sampling dropout mask
        # TODO: add drop mask
        #drop_mask = self._sample_drop_mask(w)

        wx = self.w(x)
        # Loop over time axis
        for k in range(w.shape[1]):
            gates = wx[:, k] + self.u(ht)
            it, ft, gt, ot = gates.chunk(4, dim=-1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)

            ct = ft * ct + it * gt 
            ht = ot * torch.tanh(ct)

            hiddens.append(ht)
            cell_state.append(ct)

        # Stacking states
        h = torch.stack(hiddens, dim=1)
        c = torch.stack(cell_state, dim=1)
        return h, c


if __name__ == "__main__":

    hidden_size = 5
    input_size = 5
    batch_size = 1
    seq_length=10
    ht = torch.randn(batch_size, seq_length, hidden_size)
    ct = torch.randn(batch_size, seq_length, hidden_size)
    xt = torch.randn(batch_size, input_size)
    w = nn.Linear(input_size, 4 * hidden_size, bias=True)
    u = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    lstm_layer = LSTM_Layer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_size=batch_size,
        dropout=0.0,
        bias=True,
        bidirectional=False
    )

    
