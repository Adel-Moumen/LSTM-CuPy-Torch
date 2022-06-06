import torch
import torch.autograd as autograd
import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor

try:
    import cupy as cp
except ImportError:
    err_msg = "The optional dependency CuPy is needed to use LiGRU on CuPy\n"
    err_msg += "Cannot import CuPy.\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "If you use your localhost:\n"
    err_msg += "$ python -m pip install -U setuptools pip\n"
    err_msg += "$ pip install cupy-cudaXXX (XXX is your Cuda Toolkit version)\n"
    err_msg += "If you use conda:\n"
    err_msg += "$ conda install -c conda-forge cupy"
    err_msg += "for more details: https://docs.cupy.dev/en/stable/install.html"
    err_msg += "=============================\n"
    raise ImportError(err_msg)
    

def transform_tensor_to_cupy(x):
    """Transform a PyTorch Tensor located on device="cuda" to a CuPy array. 
    
    Argument
    --------
        x : torch.Tensor
    """
    return cp.ascontiguousarray(cp.from_dlpack(torch.utils.dlpack.to_dlpack(x.detach())))

class _lstm_cell_jit(torch.nn.Module):
    """This class redefines the forward of a LiGRU cell.
    """

    def __init__(self):
        super(_lstm_cell_jit, self).__init__()

    def forward(self, wx, u, u_bias, ht, ct):
        #TODO: Add drop mask

        # Loop over time axis
        hiddens = []
        cell_state = []

        save_it = []
        save_ft = []
        save_gt = []
        save_ot = []

        for k in range(wx.shape[1]):

            gates = wx[:, k] + (ht @ u.T) + u_bias 
            it, ft, gt, ot = gates.chunk(4, dim=1)

            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)

            save_it.append(it)
            save_ft.append(ft)
            save_gt.append(gt)
            save_ot.append(ot)

            ct = ft * ct + it * gt 
            ht =  ot * torch.tanh(ct)

            hiddens.append(ht)
            cell_state.append(ct)

        # Stacking states
        h = torch.stack(hiddens, dim=1)
        c = torch.stack(cell_state, dim=1)
        it = torch.stack(save_it, dim=1)
        ft = torch.stack(save_ft, dim=1)
        gt = torch.stack(save_gt, dim=1)
        ot = torch.stack(save_ot, dim=1)

        return h, c, it, ft, gt, ot 

class _LSTM_Cell(autograd.Function):

    @staticmethod
    def forward(ctx, cell_jit, wx, u, u_bias, ht, ct):
        
        h, c, it, ft, gt, ot = cell_jit(wx, u, u_bias, ht, ct)
        ctx.save_for_backward(it, ft, gt, ot, c, h, u, wx)


        return h, c

    @staticmethod
    def backward(ctx, grad_out_h, grad_out_c):
        it, ft, gt, ot , c, h, u, wx = ctx.saved_tensors


        
        it, ft, gt, ot, c, h, u, wx, grad_out_h, grad_out_c, = (transform_tensor_to_cupy(x) 
                            for x in [it, ft, gt, ot, c, h, u, wx, grad_out_h, grad_out_c])


        dwx = cp.empty_like(wx)
        dh = cp.zeros_like(h[:, 0])
        dc = cp.zeros_like(h[:, 0])
        dh_prev, dc_prev = cp.zeros_like(h[:, 0]), cp.zeros_like(h[:, 0])
        h_init = cp.zeros_like(h[:, 0])
        c_init = cp.zeros_like(c[:, 0])
        du = cp.zeros_like(u)
        
        _preamble = '''
        
        template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
        template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }
        '''
        
        _lstm_cell_backward_kernel = cp.ElementwiseKernel(
            'T grad_out_h, T grad_out_c, T dh_prev, T dc_prev, T c, T c_prev, T g, T _i, T f, T o',
            'T dh, T dc, T dc_prev_, T di, T df, T dg, T dot',
            '''
            dh = grad_out_h + dh_prev;
            dc = grad_tanh(tanh(c)) * o * dh + dh_prev + grad_out_c;
            di = dc * g * grad_sigmoid(_i);
            df = dc * c_prev * grad_sigmoid(f);
            dg = dc * _i * grad_tanh(g);
            dot = dh * tanh(c) * grad_sigmoid(o);
            dc_prev_ = dc * f;
            ''',
            '_lstm_cell_backward_kernel', 
            preamble=_preamble
        )
        
        idx = dwx.shape[2] // 4
        
        for t in reversed(range(wx.shape[1])):
            
            ct = c_init if t - 1 < 0 else c[:, t-1]
            ht = h_init if t - 1 < 0 else h[:, t-1]
            
            _lstm_cell_backward_kernel(
                grad_out_h[:, t], 
                grad_out_c[:, t],
                dh_prev, 
                dc_prev, 
                c[:, t], 
                ct, 
                gt[:, t], 
                it[:, t], 
                ft[:, t], 
                ot[:, t],
                dh,
                dc,
                dc_prev,
                dwx[:, t, :idx], 
                dwx[:, t, idx:idx*2],
                dwx[:, t, idx*2:idx*3],
                dwx[:, t, idx*3:]
                
            )

            
            du += dwx[:, t].T.dot(ht)
            dh_prev = dwx[:, t].dot(u) 
            
               
        return None, torch.from_dlpack(dwx), torch.from_dlpack(du), torch.from_dlpack(dwx), None, None

class LSTM(torch.nn.Module):
    """ 
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        num_layers=1,
        bias=True,
        dropout=0.0,
        re_init=True,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.re_init = re_init
        self.bidirectional = bidirectional
        self.reshape = False

        # Computing the feature dimensionality
        if len(input_shape) > 3:
            self.reshape = True
        self.fea_dim = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

        if self.re_init:
            rnn_init(self.rnn)

    def _init_layers(self):
        """Initializes the layers of the liGRU."""
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = LSTM_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the liGRU.
        Arguments
        ---------
        x : torch.Tensor
            The input tensor.
        hx : torch.Tensor
            Starting hidden state.
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run lstm
        output, hh, ct = self._forward_lstm(x, hx=hx)

        return output, (hh, ct)

    def _forward_lstm(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla liGRU.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
        """
        h = []
        c = []
        if hx is not None:
            if self.bidirectional:
                hx[0] = hx[0].reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )

                hx[1] = hx[1].reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )
        # Processing the different layers
        for i, lstm_lay in enumerate(self.rnn):
            if hx is not None:
                x, ct = lstm_lay(x, hx=(hx[0][i], hx[1][i]))
            else:
                x, ct = lstm_lay(x, hx=None)
            h.append(x[:, -1, :])
            c.append(ct[:, -1, :])
        h = torch.stack(h, dim=1)
        c = torch.stack(c, dim=1)
        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
            c = c.reshape(c.shape[1] * 2, c.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)
            c = c.transpose(0, 1)

        return x, h, c



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

        self._lstm_cell_jit = torch.jit.script(_lstm_cell_jit())


    def forward(self, x, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # type: (torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor] # noqa F821
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
        
        wx = self.w(x)
        
        # Processing time steps
        if hx is not None:
            h, c, = hx 

            h, c = _LSTM_Cell.apply(self._lstm_cell_jit, wx, self.u.weight, self.u.bias, h, c)
        else:
            h, c = _LSTM_Cell.apply(self._lstm_cell_jit, wx, self.u.weight, self.u.bias, self.h_init, self.c_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

            c_f, c_b = c.chunk(2, dim=0)
            c_b = c_b.flip(1)
            c = torch.cat([c_f, c_b], dim=2)

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

def rnn_init(module):
    """This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.
    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)

if __name__ == "__main__":

    hidden_size = 20
    input_size = 20
    batch_size = 20
    seq_length = 20
    

    inp_tensor = torch.rand([batch_size, seq_length, input_size]).to("cuda")
    net = LSTM(input_shape=inp_tensor.shape, hidden_size=hidden_size).to("cuda")

    out_tensor,(h, c) = net(inp_tensor)
    
    out_tensor.sum().backward()
    
