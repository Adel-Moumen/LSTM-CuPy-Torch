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
        self.register_buffer("c_init", torch.zeros(1, self.hidden_size))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop()


    def forward(self, x, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # type: (torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor # noqa F821
        """Returns the output of the liGRU layer.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)
        
        # Processing time steps
        if hx is not None:
            h, c = hx 
            h = self._lstm_cell(w, h, c)
        else:
            h = self._lstm_cell(w, self.h_init, self.c_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h


    def _lstm_cell(self, wx: torch.Tensor, ht: torch.Tensor, ct: torch.Tensor):
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
        drop_mask = self._sample_drop_mask(wx)

        # Loop over time axis
        for k in range(wx.shape[1]):
            gates = wx[:, k] + self.u(ht)
            it, ft, gt, ot = gates.chunk(4, dim=-1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)

            ct = ft * ct + it * gt 
            ht = ot * torch.tanh(ct) * drop_mask

            hiddens.append(ht)
            cell_state.append(ct)

        # Stacking states
        h = torch.stack(hiddens, dim=1)
        c = torch.stack(cell_state, dim=1)
        return h, c

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks"""
        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=w.device
                    )
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            self.drop_mask_te = self.drop_mask_te.to(w.device)
            drop_mask = self.drop_mask_te

        return drop_mask

    def _init_drop(self):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size)).data,
        )
        self.register_buffer("drop_mask_te", torch.tensor([1.0]).float())

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=x.device,
                    )
                ).data

if __name__ == "__main__":

    hidden_size = 5
    input_size = 5
    batch_size = 1
    seq_length=10
    
    x = torch.randn(batch_size, seq_length, input_size)
    lstm_layer = LSTM_Layer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_size=batch_size,
        dropout=0.0,
        bias=True,
        bidirectional=False
    )

    out = lstm_layer(x)
    print(out)
    
