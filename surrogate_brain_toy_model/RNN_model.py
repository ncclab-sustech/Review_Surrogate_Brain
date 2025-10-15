import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import math
from typing import Tuple
from torch.linalg import pinv
from typing import Optional, Tuple


#-----------------------------Low-Rank RNN--------------------------------------------

class lr_RNN(nn.Module):
    """
    Low-Rank RNN  
    Costacurta J, Bhandarkar S, Zoltowski D, et al. Structured flexibility in recurrent neural networks via neuromodulation[J]. 
    Advances in Neural Information Processing Systems, 2024, 37: 1954-1972.

    Args:
        in_dim     : Input feature dimension P
        x_dim      : Main state dimension K
        rank       : Low-rank factor R
        nm_dim     : Neuromodulator state dimension M
        tau_x      : Time constant of main state (larger = slower dynamics)
        tau_z      : Time constant of neuromodulator state
        tau        : Teacher forcing interval (update with ground-truth input every tau steps)
    """
    def __init__(
        self,
        in_dim: int,
        x_dim: int,
        rank: int,
        nm_dim: int,
        *,
        tau_x: float = 100.0,
        tau_z: float = 100.0,
        tau: int = 5,
    ):
        super().__init__()
        self.in_dim     = in_dim
        self.x_dim      = x_dim
        self.rank       = rank
        self.nm_dim     = nm_dim
        self.tau_x      = tau_x
        self.tau_z      = tau_z
        self.tau        = tau  # teacher forcing interval

        # ---------- Low-rank transition parameters ----------
        self.L = nn.Parameter(torch.randn(x_dim, rank) * 0.1)
        self.R = nn.Parameter(torch.randn(x_dim, rank) * 0.1)

        # ---------- Neuromodulator subnetwork ----------
        self.Wz = nn.Parameter(torch.randn(nm_dim, nm_dim) * 0.1)
        self.Az = nn.Linear(nm_dim, rank)

        # ---------- Encoder networks ----------
        self.encoder_x = nn.Sequential(
            nn.Linear(in_dim, x_dim), nn.Tanh(),
            nn.Linear(x_dim, x_dim), nn.Tanh(),
            nn.Linear(x_dim, x_dim), nn.Tanh()
        )
        self.encoder_z = nn.Sequential(
            nn.Linear(in_dim, nm_dim), nn.Tanh(),
            nn.Linear(nm_dim, nm_dim), nn.Tanh(),
            nn.Linear(nm_dim, nm_dim), nn.Tanh()
        )

        # ---------- Decoder network ----------
        self.decoder = nn.Sequential(
            nn.Linear(x_dim, x_dim), nn.Tanh(),
            nn.Linear(x_dim, in_dim), nn.Tanh()
        )

    def forward(
        self,
        in_seq: torch.Tensor,                     # (B, T, P)   Input sequence
        *,
        teacher_forcing: bool = True,
        return_s: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        B, T, _ = in_seq.shape
        device  = in_seq.device

        # -------- Encode input sequence --------
        x_seq = self.encoder_x(in_seq)    # (B, T, K)
        z_seq = self.encoder_z(in_seq)    # (B, T, M)

        # -------- Initialize hidden states --------
        x = x_seq[:, :1, :]               # (B, 1, K)   initial main state
        z = z_seq[:, :1, :]               # (B, 1, M)   initial neuromodulator state

        # -------- Allocate output tensors --------
        x_pred     = torch.zeros(B, T, self.in_dim,  device=device)   # reconstructed input
        z_pred     = torch.zeros(B, T, self.nm_dim,  device=device)   # neuromodulator trajectory
        s_seq_pred = torch.zeros(B, T, self.rank,    device=device) if return_s else None  # modulatory weights

        # -------- Main loop over time steps --------
        for t in range(T):
            # ---------- Teacher forcing ----------
            if teacher_forcing and t < T and self.tau > 0 and t % self.tau == 0 and t > 0:
                # Replace current state with encoded ground-truth every tau steps
                x = x_seq[:, t:t+1, :]
                z = z_seq[:, t:t+1, :]

            # ---------- Update neuromodulator state z ----------
            phi_z = torch.tanh(z)
            dz    = -z + phi_z @ self.Wz.T
            z     = z + (1.0 / self.tau_z) * dz
            z_pred[:, t:t+1, :] = z

            # ---------- Modulate low-rank factor s ----------
            s = torch.sigmoid(self.Az(z))                      # (B, 1, R)
            if return_s:
                s_seq_pred[:, t:t+1, :] = s

            # ---------- Construct effective weight matrix W ----------
            # einsum builds: W_b = L * diag(s_b) * Rᵀ → (B, K, K)
            W = torch.einsum('bir,jr,kr->bijk', s, self.L, self.R)[:, 0]  # (B, K, K)

            # ---------- Update main state x ----------
            dx = -x + torch.bmm(torch.tanh(x), W)              # (B, 1, K)
            x  = x + (1.0 / self.tau_x) * dx
            x_pred[:, t:t+1, :] = self.decoder(x)              # decode back to input space (B, 1, P)

        # -------- Reconstruction of encoded sequence (for training loss) --------
        x_recon = self.decoder(x_seq)

        return x_pred, x_recon, z_pred, z_seq


#-----------------------------EI RNN--------------------------------------------


class PosWLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that the weight matrix is constrained to be non-negative.
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(PosWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.with_Tanh = kwargs.get('with_Tanh', False)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization with gain for 'tanh'
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Enforce non-negativity on weights
        output = F.linear(input, torch.abs(self.weight), self.bias)
        if self.with_Tanh:
            # Alternative: enforce positivity with ReLU
            output = F.linear(input, F.relu(self.weight), self.bias)
        return output
    
    
class EIRecLinear(nn.Module):
    r"""
    Recurrent E-I linear transformation with flexible Dale's law and block-structure constraints.

    Args:
        hidden_size: int, total number of hidden neurons
        e_prop: float in (0,1), proportion of excitatory units
        mode: str, 'none', 'dense', or 'block' (defines how Dale's law is applied)
        block_groups: int, number of groups when mode='block'
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop, mode='none', block_groups=2, bias=True, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.mode = mode
        self.block_groups = block_groups

        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        
        # Dale's law mask: excitatory (positive), inhibitory (negative)
        if self.mode in ['dense', 'block']:
            mask = np.tile([1]*self.e_size + [-1]*self.i_size, (hidden_size, 1))
            np.fill_diagonal(mask, 0)  # no self-connections
            self.register_buffer('dale_mask', torch.tensor(mask, dtype=torch.float32))
        else:
            mask = np.ones((hidden_size, hidden_size))
            np.fill_diagonal(mask, 0)
            self.register_buffer('dale_mask', torch.tensor(mask, dtype=torch.float32))

        # Block-structured connectivity mask
        if self.mode == 'block':
            group_size = self.e_size // self.block_groups
            block_mask = torch.ones(hidden_size, hidden_size)

            for g in range(self.block_groups):
                start = g * group_size
                end = (g+1) * group_size if g < self.block_groups-1 else self.e_size
                block_mask[start:end, :start] = 0
                block_mask[start:end, end:self.e_size] = 0
            # Allow all inhibitory connections
            block_mask[:, self.i_size+self.e_size:] = 1
            block_mask[self.e_size:, :] = 1
            self.register_buffer('block_mask', block_mask)
        else:
            self.register_buffer('block_mask', torch.ones(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.mode in ['dense', 'block']:
            # Scale excitatory columns by E/I ratio
            self.weight.data[:, :self.e_size] /= (self.e_size / self.i_size)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        if self.mode == 'none':
            return self.weight
        else:
            return torch.abs(self.weight) * self.dale_mask * self.block_mask

    def forward(self, input):
        return F.linear(input, self.effective_weight(), self.bias)
    

class EIRNN(nn.Module):
    """
    Excitatory-Inhibitory RNN with Dale's law constraints.  
    Reference: Song et al. (2016), PLoS Comput Biol.

    Args:
        input_size: input dimension
        hidden_size: total number of hidden neurons
        e_prop: proportion of excitatory neurons (0–1)
        dt: integration time step; if None, use default alpha=0.3
        sigma_rec: standard deviation of recurrent noise
    """
    def __init__(self, input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size * e_prop)
        self.i_size = hidden_size - self.e_size
        self.tau = 200

        self.mode = kwargs.get('mode', 'none')
        self.noneg = kwargs.get('noneg', True)
        self.with_Tanh = kwargs.get('with_Tanh', False)

        # Time-step dynamics
        self.alpha = dt / self.tau if dt else 0.3
        self.oneminusalpha = 1 - self.alpha
        self._sigma_rec = np.sqrt(2 * self.alpha) * sigma_rec

        # Input-to-hidden connection (optionally constrained to positive weights)
        if self.noneg and self.mode != 'none':
            self.input2h = PosWLinear(input_size, hidden_size, with_Tanh=self.with_Tanh)
        else:
            self.input2h = nn.Linear(input_size, hidden_size)

        # Hidden-to-hidden recurrent connections (supports none/dense/block modes)
        self.h2h = EIRecLinear(hidden_size, e_prop=e_prop, mode=self.mode)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        device = input.device
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))

    def recurrence(self, input, hidden):
        state, output = hidden
        total_input = self.h2h(output)
        state = state * self.oneminusalpha + total_input * self.alpha
        state += self._sigma_rec * torch.randn_like(state)  # recurrent noise
        output = torch.tanh(state)
        return state, output

    def forward(self, input, hidden=None):
        """
        Args:
            input: (T, B, P)
        Returns:
            outputs: (T, B, H)
            final hidden state: (state, output)
        """
        if hidden is None:
            hidden = self.init_hidden(input)

        outputs = []
        for t in range(input.size(0)):
            hidden = self.recurrence(input[t], hidden)
            outputs.append(hidden[1])  # collect outputs
        return torch.stack(outputs, dim=0), hidden


class Net(nn.Module):
    """
    High-level wrapper: encoder + EIRNN + decoder.

    Args:
        input_size: input dimension
        hidden_size: hidden dimension
        output_size: output dimension
    """
    def __init__(self, input_size, hidden_size, output_size, tau=5, **kwargs):
        super().__init__()
        self.tau = tau
        self.mode = kwargs.get('mode', 'none')
        self.noneg = kwargs.get('noneg', True)
        self.with_Tanh = kwargs.get('with_Tanh', False)

        self.rnn = EIRNN(input_size, hidden_size, **kwargs)

        # Input encoder (multi-layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

        # Output decoder (optionally using only excitatory neurons)
        if self.noneg and self.mode != 'none':
            self.fc = PosWLinear(self.rnn.e_size, output_size, with_Tanh=self.with_Tanh)
        elif self.mode == 'none':
            self.fc = nn.Linear(self.rnn.hidden_size, output_size)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn.hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, output_size), nn.Tanh()
            )

    def _decode(self, h):
        return self.fc(h)

    def forward(self, in_seq, teacher_forcing=True):
        """
        Args:
            in_seq: (B, T, P)
        Returns:
            x_pred: predicted sequence (B, T, P)
            x_recon: reconstruction (B, T, P)
        """
        B, T, P = in_seq.shape
        device = in_seq.device

        x_pred = torch.zeros(B, T, P, device=device)
        in_seq_enc = self.encoder(in_seq)                         # Encode input (B, T, H)
        rnn_out_teacher, _ = self.rnn(in_seq_enc.transpose(0, 1)) # Teacher-forced RNN output (T, B, H)
        x_recon = self._decode(rnn_out_teacher).transpose(0, 1)   # Reconstruction (B, T, P)

        # Autoregressive prediction
        y_seq = []
        state = in_seq_enc[:, 0, :]
        output = torch.tanh(state)

        for t in range(T):
            if teacher_forcing and t < T and t > 0 and t % self.tau == 0:
                state = in_seq_enc[:, t, :]
            elif t == 0:
                state = in_seq_enc[:, 0, :]
                output = torch.tanh(state)

            state, output = self.rnn.recurrence(0, (state, output))
            y_t = self._decode(output)
            y_seq.append(y_t)

        x_pred = torch.stack(y_seq, dim=1)  # (B, T, P)
        return x_pred, x_recon


class PosWLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that the weight matrix is constrained to be non-negative.
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(PosWLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.with_Tanh = kwargs.get('with_Tanh', False)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization with gain for 'tanh'
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Enforce non-negativity on weights
        output = F.linear(input, torch.abs(self.weight), self.bias)
        if self.with_Tanh:
            # Alternative: enforce positivity with ReLU
            output = F.linear(input, F.relu(self.weight), self.bias)
        return output
    
    
class EIRecLinear(nn.Module):
    r"""
    Recurrent E-I linear transformation with flexible Dale's law and block-structure constraints.

    Args:
        hidden_size: int, total number of hidden neurons
        e_prop: float in (0,1), proportion of excitatory units
        mode: str, 'none', 'dense', or 'block' (defines how Dale's law is applied)
        block_groups: int, number of groups when mode='block'
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop, mode='none', block_groups=2, bias=True, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.mode = mode
        self.block_groups = block_groups

        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        
        # Dale's law mask: excitatory (positive), inhibitory (negative)
        if self.mode in ['dense', 'block']:
            mask = np.tile([1]*self.e_size + [-1]*self.i_size, (hidden_size, 1))
            np.fill_diagonal(mask, 0)  # no self-connections
            self.register_buffer('dale_mask', torch.tensor(mask, dtype=torch.float32))
        else:
            mask = np.ones((hidden_size, hidden_size))
            np.fill_diagonal(mask, 0)
            self.register_buffer('dale_mask', torch.tensor(mask, dtype=torch.float32))

        # Block-structured connectivity mask
        if self.mode == 'block':
            group_size = self.e_size // self.block_groups
            block_mask = torch.ones(hidden_size, hidden_size)

            for g in range(self.block_groups):
                start = g * group_size
                end = (g+1) * group_size if g < self.block_groups-1 else self.e_size
                block_mask[start:end, :start] = 0
                block_mask[start:end, end:self.e_size] = 0
            # Allow all inhibitory connections
            block_mask[:, self.i_size+self.e_size:] = 1
            block_mask[self.e_size:, :] = 1
            self.register_buffer('block_mask', block_mask)
        else:
            self.register_buffer('block_mask', torch.ones(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.mode in ['dense', 'block']:
            # Scale excitatory columns by E/I ratio
            self.weight.data[:, :self.e_size] /= (self.e_size / self.i_size)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        if self.mode == 'none':
            return self.weight
        else:
            return torch.abs(self.weight) * self.dale_mask * self.block_mask

    def forward(self, input):
        return F.linear(input, self.effective_weight(), self.bias)
    

class EIRNN(nn.Module):
    """
    Excitatory-Inhibitory RNN with Dale's law constraints.  
    Reference: Song et al. (2016), PLoS Comput Biol.

    Args:
        input_size: input dimension
        hidden_size: total number of hidden neurons
        e_prop: proportion of excitatory neurons (0–1)
        dt: integration time step; if None, use default alpha=0.3
        sigma_rec: standard deviation of recurrent noise
    """
    def __init__(self, input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size * e_prop)
        self.i_size = hidden_size - self.e_size
        self.tau = 200

        self.mode = kwargs.get('mode', 'none')
        self.noneg = kwargs.get('noneg', True)
        self.with_Tanh = kwargs.get('with_Tanh', False)

        # Time-step dynamics
        self.alpha = dt / self.tau if dt else 0.3
        self.oneminusalpha = 1 - self.alpha
        self._sigma_rec = np.sqrt(2 * self.alpha) * sigma_rec

        # Input-to-hidden connection (optionally constrained to positive weights)
        if self.noneg and self.mode != 'none':
            self.input2h = PosWLinear(input_size, hidden_size, with_Tanh=self.with_Tanh)
        else:
            self.input2h = nn.Linear(input_size, hidden_size)

        # Hidden-to-hidden recurrent connections (supports none/dense/block modes)
        self.h2h = EIRecLinear(hidden_size, e_prop=e_prop, mode=self.mode)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        device = input.device
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))

    def recurrence(self, input, hidden):
        state, output = hidden
        total_input = self.h2h(output)
        state = state * self.oneminusalpha + total_input * self.alpha
        state += self._sigma_rec * torch.randn_like(state)  # recurrent noise
        output = torch.tanh(state)
        return state, output

    def forward(self, input, hidden=None):
        """
        Args:
            input: (T, B, P)
        Returns:
            outputs: (T, B, H)
            final hidden state: (state, output)
        """
        if hidden is None:
            hidden = self.init_hidden(input)

        outputs = []
        for t in range(input.size(0)):
            hidden = self.recurrence(input[t], hidden)
            outputs.append(hidden[1])  # collect outputs
        return torch.stack(outputs, dim=0), hidden


class Net(nn.Module):
    """
    High-level wrapper: encoder + EIRNN + decoder.

    Args:
        input_size: input dimension
        hidden_size: hidden dimension
        output_size: output dimension
    """
    def __init__(self, input_size, hidden_size, output_size, tau=5, **kwargs):
        super().__init__()
        self.tau = tau
        self.mode = kwargs.get('mode', 'none')
        self.noneg = kwargs.get('noneg', True)
        self.with_Tanh = kwargs.get('with_Tanh', False)

        self.rnn = EIRNN(input_size, hidden_size, **kwargs)

        # Input encoder (multi-layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

        # Output decoder (optionally using only excitatory neurons)
        if self.noneg and self.mode != 'none':
            self.fc = PosWLinear(self.rnn.e_size, output_size, with_Tanh=self.with_Tanh)
        elif self.mode == 'none':
            self.fc = nn.Linear(self.rnn.hidden_size, output_size)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.rnn.hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, output_size), nn.Tanh()
            )

    def _decode(self, h):
        return self.fc(h)

    def forward(self, in_seq, teacher_forcing=True):
        """
        Args:
            in_seq: (B, T, P)
        Returns:
            x_pred: predicted sequence (B, T, P)
            x_recon: reconstruction (B, T, P)
        """
        B, T, P = in_seq.shape
        device = in_seq.device

        x_pred = torch.zeros(B, T, P, device=device)
        in_seq_enc = self.encoder(in_seq)                         # Encode input (B, T, H)
        rnn_out_teacher, _ = self.rnn(in_seq_enc.transpose(0, 1)) # Teacher-forced RNN output (T, B, H)
        x_recon = self._decode(rnn_out_teacher).transpose(0, 1)   # Reconstruction (B, T, P)

        # Autoregressive prediction
        y_seq = []
        state = in_seq_enc[:, 0, :]
        output = torch.tanh(state)

        for t in range(T):
            if teacher_forcing and t < T and t > 0 and t % self.tau == 0:
                state = in_seq_enc[:, t, :]
            elif t == 0:
                state = in_seq_enc[:, 0, :]
                output = torch.tanh(state)

            state, output = self.rnn.recurrence(0, (state, output))
            y_t = self._decode(output)
            y_seq.append(y_t)

        x_pred = torch.stack(y_seq, dim=1)  # (B, T, P)
        return x_pred, x_recon


#-----------------------------dend-PLRNN--------------------------------------------
class PLRNN(nn.Module):
    """
    Dendr-PLRNN (Durstewitz 2017): Latent variable dynamical system 
    using piecewise linear basis expansions.

    Args:
        dim_x: Input dimension (observation space)
        dim_z: Latent dimension (hidden state)
        n_bases: Number of basis functions
        clip_range: Optional clipping range for latent states
        mean_centering: Whether to apply mean-centering normalization
        dataset: Used for initializing basis centers (thetas) from data range
        tau: Teacher forcing interval (steps between using ground-truth inputs)
    """
    def __init__(self, dim_x: int, dim_z: int, n_bases: int,
                 clip_range: Optional[float], mean_centering: bool, dataset=None, tau: Optional[int] = None):
        super().__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.n_bases = n_bases
        self.tau = tau

        # Encoder: maps observations x into latent space z
        self.encoder = nn.Sequential(
            nn.Linear(dim_x, dim_z), nn.Tanh(),
            nn.Linear(dim_z, dim_z), nn.Tanh()
        )

        # Decoder: reconstructs x back from latent state z
        self.decoder = nn.Sequential(
            nn.Linear(dim_z, dim_z), nn.Tanh(),
            nn.Linear(dim_z, dim_x), nn.Tanh()
        )

        # Latent dynamics step module
        self.latent_step = PLRNN_Basis_Step(
            db=n_bases,
            dz=dim_z,
            clip_range=clip_range,
            layer_norm=mean_centering,
            dataset=dataset
        )

    def get_parameters(self):
        """Return dynamical system parameters (A, W, h, alphas, thetas)."""
        return (
            torch.diag(self.latent_step.AW),                         # A: diagonal elements
            self.latent_step.AW - torch.diag_embed(torch.diag(self.latent_step.AW)),  # W: off-diagonal
            self.latent_step.h,
            self.latent_step.alphas,
            self.latent_step.thetas
        )

    def teacher_force(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply teacher forcing by encoding the true observation x into latent space z."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor,
                z0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional teacher forcing.
        Every `tau` steps, the latent state z is reset using the ground-truth input.

        Args:
            x: Input sequence (B, T, P)
            z0: Optional initial latent state (B, d_z)

        Returns:
            x_pred: Reconstructed observations (B, T, P)
            z_seq: Latent state sequence (B, T, d_z)
        """
        x_ = x.permute(1, 0, 2)           # (T, B, P)
        T, B_size, _ = x_.shape

        if z0 is None:
            z = torch.randn(B_size, self.d_z, device=x.device)
            z = self.teacher_force(z, x_[0])
        else:
            z = z0

        z_seq, x_pred = [], []
        params = self.get_parameters()
        
        for t in range(T):
            # Apply teacher forcing every tau steps
            if t % self.tau == 0 and t > 0:
                z = self.teacher_force(z, x_[t])
            z = self.latent_step(z, *params)
            z_seq.append(z)
            x_pred.append(self.decoder(z))

        x_pred = torch.stack(x_pred, dim=0).permute(1, 0, 2)
        z_seq  = torch.stack(z_seq, dim=0).permute(1, 0, 2)
        return x_pred, z_seq

    def generate(self, T: int, data: torch.Tensor,
                 z0: Optional[torch.Tensor] = None,
                 B: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a latent trajectory of length T, starting from z0 or initialized from data[0].

        Args:
            T: Sequence length to generate
            data: Observed sequence (used for initialization)
            z0: Optional initial latent state
            B: Optional projection matrix (for linear transform)

        Returns:
            Latent sequence Z of shape (T, d_z)
        """
        B_PI = pinv(B) if B is not None else None
        if z0 is None:
            z = self.teacher_force(torch.randn(1, self.d_z, device=data.device), data[0], B_PI)
        else:
            z = z0

        Z = torch.empty((T, 1, self.d_z), device=data.device)
        Z[0] = z
        params = self.get_parameters()

        for t in range(1, T):
            Z[t] = self.latent_step(Z[t - 1], *params)

        return Z.squeeze(1)  # (T, d_z)


class PLRNN_Basis_Step(nn.Module):
    """
    Latent dynamics update:
        z_{t+1} = A * z_t + Σ (alpha · ReLU(z_t + theta)) · W^T + h
    """

    def __init__(self, db: int, dz: int, clip_range: Optional[float],
                 layer_norm: bool, dataset=None):
        super().__init__()
        self.dz = dz
        self.db = db
        self.clip_range = clip_range

        self.AW = self._init_AW()
        self.h = self._init_uniform((dz,))
        self.alphas = self._init_uniform((db,))
        self.thetas = self._init_thetas_uniform(dataset) if dataset is not None \
                      else nn.Parameter(torch.randn(dz, db))

        if layer_norm:
            # Normalize latent states by subtracting their mean
            self.norm = lambda z: z - z.mean(dim=1, keepdim=True)
        else:
            self.norm = nn.Identity()

    def _init_uniform(self, shape: Tuple[int]) -> nn.Parameter:
        """Initialize parameters uniformly within [-1/sqrt(n), 1/sqrt(n)]."""
        r = 1 / (shape[0] ** 0.5)
        tensor = torch.empty(*shape)
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor)

    def _init_AW(self) -> nn.Parameter:
        '''
        Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network
        with ReLU Nonlinearity https://arxiv.org/abs/1511.03771.
        '''
        rand = torch.randn(self.dz, self.dz)
        positive_def = (1 / self.dz) * rand.T @ rand
        mat = torch.eye(self.dz) + positive_def
        max_ev = torch.linalg.eigvals(mat).abs().max()
        matrix_spectral_norm_one = mat / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True) #nn.Parameter(mat / max_ev)

    def _init_thetas_uniform(self, dataset) -> nn.Parameter:
        '''
        Initialize theta matrix of the basis expansion models such that 
        basis thresholds are uniformly covering the range of the given dataset
        '''
        mn, mx = dataset.data.min().item(), dataset.data.max().item()
        theta = torch.empty((self.dz, self.db))
        nn.init.uniform_(theta, -mx, -mn)
        return nn.Parameter(theta)

    def clip_z_to_range(self, z):
        """Optionally clip latent states to [-clip_range, clip_range]."""
        if self.clip_range is not None:
            return torch.clamp(z, -self.clip_range, self.clip_range)
        return z

    def forward(self, z, A, W, h, alphas, thetas):
        """
        One latent dynamics step.

        Args:
            z: Current latent state (B, dz)
            A, W, h, alphas, thetas: system parameters

        Returns:
            z_next: Updated latent state (B, dz)
        """
        z_norm = self.norm(z).unsqueeze(-1)                        # (B, dz, 1)
        basis_exp = torch.sum(alphas * torch.relu(z_norm + thetas), dim=-1)  # (B, dz)
        z_next = A * z + basis_exp @ W.T + h
        return self.clip_z_to_range(z_next)


#--------------------RNN--------------------------------------------------
class PlainRNN(nn.Module):
    """
    Plain RNN dynamical system:
    dx/dt = -x + tanh(x) @ W   (Euler discretization)

    Features:
    - τ_x controls the speed of state change
    - Teacher forcing: refreshes the hidden state with ground-truth input every τ steps
    """
    def __init__(self, in_dim: int, hidden_dim: int,
                 tau_x: float = 200.0, tau: int = 5):
        super().__init__()
        self.in_dim     = in_dim
        self.hidden_dim = hidden_dim
        self.tau_x      = tau_x
        self.tau        = tau

        # Encoder: maps input space → hidden state space
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

        # Decoder: maps hidden state space → input space (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, in_dim), nn.Tanh()
        )

        # Recurrent weight matrix W (defines latent dynamics)
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

    def forward(
        self,
        in_seq: torch.Tensor,               # Input sequence (B, T, P)
        *,
        teacher_forcing: bool = True,       # Whether to use ground-truth inputs for state refresh
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = in_seq.shape
        device = in_seq.device

        # -------- Encode observations --------
        x_seq = self.encoder(in_seq)        # Encoded sequence in hidden space (B, T, H)

        # -------- Initialize hidden state & output buffer --------
        x = x_seq[:, :1, :]                 # Initial hidden state at t=0 (B, 1, H)
        # Alternative: encode only the first input → x = self.encoder(in_seq[:, :1, :])
        x_pred = torch.zeros(B, T, self.in_dim, device=device)

        # -------- Dynamical loop --------
        for t in range(T):
            # Teacher forcing: refresh hidden state every τ steps using true input
            if teacher_forcing and (t < T) and (t % self.tau == 0) and (t > 0):
                x = x_seq[:, t:t+1, :]      # Replace hidden state with encoder output

            dx = -x + torch.tanh(x) @ self.W.T
            x = x + dx / self.tau_x         # Euler integration step
            x_pred[:, t:t+1, :] = self.decoder(x)  # Decode back to input space

        # Optional: reconstruct the entire input sequence (useful for self-supervised training)
        x_recon = self.decoder(x_seq)       # (B, T, P)
        return x_pred, x_recon

