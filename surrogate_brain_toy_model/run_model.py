from dataprocess import build_loaders,build_loo_loaders
import numpy as np
from RNN_model import lr_RNN, Net, PLRNN, PlainRNN
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from evaluate import regression_metrics
import matplotlib.pyplot as plt
import pandas as pd
import copy
import torch, torch.nn.functional as F
from tqdm import tqdm

def build_model(model_type, dim_x, dim_z, tau, device, n_bases=10,dataset=None,rank=10,nm_dim=64):
    if model_type == 'dendPLRNN':
        model = PLRNN(
            dim_x, dim_z,
            n_bases=n_bases,
            clip_range=None,
            mean_centering=False,
            dataset=dataset,
            tau=tau
        )
    elif model_type =='EIRNN':
        model = Net(input_size=dim_x, 
                    hidden_size=dim_z, 
                    output_size=dim_x, 
                    sigma_rec=0.15,
                    mode='dense',
                    noneg=False, 
                    with_Tanh=True,
                    tau_x=200,
                    tau=tau)
        
    elif model_type =='lrRNN':
        model = lr_RNN(
            in_dim=dim_x,
            x_dim=dim_z,
            rank=rank,
            nm_dim=nm_dim,
            tau_x=200,
            tau_z=200,
            tau=tau,                  
        )
        
    elif model_type=='RNN':
        model = PlainRNN(
            in_dim=dim_x,
            hidden_dim=dim_z,
            tau_x=200,
            tau=tau,                  
        )   
        
    return model.to(device)

def val_loss_compute(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            hist = batch[0].to(device)
            x_pred, *_ = model(hist)

            loss = criterion(hist[:, 1:, :], x_pred[:, 1:, :])

            total_loss += loss.item() * hist.size(0)

    return total_loss / len(loader.dataset)



def train_model(model, train_loader, val_loader,
                criterion, optimizer, scheduler, early_stop,
                num_epochs, save_path, regularizers=None,
                device='cpu', dim_z=512, tau=5, model_type='RNN'):
    """
    Train a model with optional regularizers and early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler (ReduceLROnPlateau, etc.)
        early_stop: EarlyStopping instance
        num_epochs: maximum number of epochs
        save_path: path to save the best model
        regularizers: dict[name → (reg_fn, weight)], optional
        device: training device ("cpu" or "cuda")
        dim_z, tau, model_type: metadata stored in the checkpoint
    """
    best_val = float("inf")
    best_model_state = None
    best_optimizer_state = None
    best_epoch = -1
    
    if regularizers is None:
        regularizers = {}  # no regularization if not provided

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            seq = batch[0].to(device)
            optimizer.zero_grad()

            x_pred, *_ = model(seq)

            # Reconstruction loss: predict next step given current sequence
            recon_pred = x_pred[:, :-1, :]
            recon_true = seq[:, 1:, :]
            loss = criterion(recon_pred, recon_true)
            
            # Apply optional regularizers
            for name, (reg_fn, lam) in regularizers.items():
                if lam:  # skip if weight = 0
                    reg_val = reg_fn(model=model,
                                     recon_pred=recon_pred,
                                     recon_true=recon_true,
                                     batch=batch)
                    loss = loss + lam * reg_val
    
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * seq.size(0)

        # Compute epoch-level metrics
        train_loss = total_loss / len(train_loader.dataset)
        val_loss = val_loss_compute(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d}: train={train_loss:.4f}, val={val_loss:.4f}")

        scheduler.step(val_loss)
        early_stop.step(val_loss)

        # Track best validation loss
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

        if early_stop.should_stop:
            print("Early stopping.")
            break

    # Save best checkpoint
    if best_model_state is not None:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'val_loss': best_val,
            'regularizers': regularizers,
            'dim_z': dim_z,
            'tau': tau,
            'model_type': model_type,  # store model type as metadata
        }, save_path)
        print(f"Best model saved at epoch {best_epoch}: {save_path}")
    else:
        print("Warning: No improvement during training, model not saved.")

    return save_path



def evaluate_model(model, test_loader, ckpt_path, tau, device, name, model_type, reg_tag="", dim_z=512):
    """
    Evaluate a trained model on the test set, save metrics, predictions, and plots.

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        ckpt_path: path to the saved model checkpoint (.pth)
        tau: prediction horizon (steps)
        device: device to run evaluation ("cpu" or "cuda")
        name: experiment/patient identifier
        model_type: string describing model type (e.g., "RNN")
        reg_tag: optional string describing regularizers used
        dim_z: latent dimension size

    Returns:
        None (saves evaluation metrics, predictions, and preview plots to disk)
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create directories for results and metrics
    root = Path(ckpt_path).parent
    results_dir = root / "results"
    metrics_dir = root /  "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # File paths for saving outputs
    csv_path = metrics_dir / f"{model_type}_summary_tau{tau}_hidden{dim_z}_{reg_tag}.csv"
    npz_path = results_dir / f"{model_type}_pred_tau{tau}_hidden{dim_z}_{reg_tag}.npz"
    png_path = results_dir / f"{model_type}_preview_tau{tau}_hidden{dim_z}_{reg_tag}.png"

    summary_rows = []   # store metrics per epoch
    all_preds, all_truths = [], []

    with torch.no_grad():
        for epoch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating each epoch")):
            seq = batch[0].to(device)  # input sequence (B, T, C)
            x_pred, *_ = model(seq)

            # Reshape both truth and prediction to (C, T)
            yt = seq.permute(2, 0, 1).reshape(seq.shape[2], -1).cpu().numpy()
            yp = x_pred.permute(2, 0, 1).reshape(x_pred.shape[2], -1).cpu().numpy()

            # Compute regression metrics
            df = regression_metrics(yt, yp)
            mean_row = df.loc['mean'].to_dict()

            # Add metadata for this epoch
            mean_row['epoch'] = epoch_idx
            mean_row['fc_sim'] = df.attrs.get('fc_sim', np.nan)
            mean_row['h1_wass_dist'] = df.attrs.get('h1_wass_dist', np.nan)
            summary_rows.append(mean_row)

            # Collect predictions and ground truth
            all_preds.append(yp)
            all_truths.append(yt)

    # Concatenate metrics across all epochs and save to CSV
    summary_df = pd.DataFrame(summary_rows).set_index('epoch')
    summary_df.to_csv(csv_path)

    # Save full prediction and ground truth arrays for further analysis
    pred_all = np.concatenate(all_preds, axis=1)   # shape: (channels, total_time)
    truth_all = np.concatenate(all_truths, axis=1)
    np.savez(npz_path, pred=pred_all, truth=truth_all)

    # Plotting: visualize predictions for the first 20 channels
    n_ch = 20
    truth_plot = truth_all[10:10+n_ch]
    pred_plot = pred_all[10:10+n_ch]
    channel_ranges = truth_plot.max(axis=1) - truth_plot.min(axis=1)
    y_gap = channel_ranges.max() * 0.3  # vertical offset between channels

    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    for ch in range(n_ch):
        offset = ch * y_gap
        ax.plot(truth_plot[ch] + offset, lw=1.2, color='orange',
                label='Ground truth' if ch == 0 else None)
        ax.plot(pred_plot[ch] + offset, lw=0.9, color='black',
                label='Prediction' if ch == 0 else None)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Channel value')
    ax.set_title(f'{model_type} prediction vs. ground truth')
    ax.set_yticks(np.arange(n_ch) * y_gap)
    ax.set_yticklabels([f'ch {ch}' for ch in range(n_ch)])
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.show()


# ── Weight Regularizers ────────────────────────────────────────────────
def reg_l1_weight(model, recon_pred=None, **kwargs):
    """L1 weight penalty (sparsity): ∑|W| over all weight matrices."""
    reg = torch.tensor(0., device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:  # skip bias and LN/BN params
            reg += p.abs().sum()
    return reg


def reg_l2_weight(model, recon_pred=None, **kwargs):
    """L2 weight penalty: ∑‖W‖² (useful if not already in optimizer.weight_decay)."""
    reg = torch.tensor(0., device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            reg += p.pow(2).sum()
    return reg


# ── Temporal Smoothness Regularizers ───────────────────────────────────
def reg_tv(model, recon_pred, tau=1, **kwargs):
    """Total Variation penalty: ‖x_t − x_{t−τ}‖₁."""
    if recon_pred.size(1) <= tau:  # handle case τ > sequence length
        return recon_pred.sum() * 0
    diff = recon_pred[:, tau:] - recon_pred[:, :-tau]
    return diff.abs().sum()


def reg_smooth(model, recon_pred, tau=1, **kwargs):
    """Smoothness penalty: ‖x_t − x_{t−τ}‖₂²."""
    if recon_pred.size(1) <= tau:
        return recon_pred.sum() * 0
    diff = recon_pred[:, tau:] - recon_pred[:, :-tau]
    return diff.pow(2).sum()


def reg_activation_l1(model, recon_pred, **kwargs):
    """L1 penalty on activations (encourages sparse outputs)."""
    return recon_pred.abs().sum()


# ── Spectral Regularizers ──────────────────────────────────────────────
def _stft_mag(x, n_fft=64, hop=None, win=None):
    """
    Compute STFT magnitude spectrum.

    Args:
        x: input tensor of shape (B, T, C)
    Returns:
        magnitude spectrogram of shape (B, C, Freq, Frame)
    """
    B, T, C = x.shape
    xc = x.permute(0, 2, 1).reshape(B * C, T)  # (B*C, T)
    spec = torch.stft(
        xc,
        n_fft=n_fft,
        hop_length=hop or n_fft // 2,
        win_length=win or n_fft,
        window=torch.hann_window(win or n_fft, device=x.device),
        return_complex=True,
        center=True,
        pad_mode="reflect"
    )  # (B*C, Freq, Frm)
    mag = spec.abs().view(B, C, *spec.shape[1:])
    return mag


def make_reg_spectral(n_fft=64, p=1, use_sc=False):
    """
    Create a spectral-domain regularizer.

    Returns:
        reg_fn(model, recon_pred, recon_true, **kwargs) → scalar Tensor

    Args:
        n_fft: FFT length (frequency resolution)
        p: 1 = L1, 2 = L2 (only when use_sc=False)
        use_sc: True = Spectral Convergence loss
    """
    def _reg(model, recon_pred, recon_true=None, **kwargs):
        if recon_true is None:
            raise ValueError("Spectral regularizer requires recon_true.")

        # truncate if sequence length < n_fft
        n = min(n_fft, recon_pred.shape[1])

        mag_p = _stft_mag(recon_pred, n_fft=n)
        mag_t = _stft_mag(recon_true,  n_fft=n)

        if use_sc:  # Spectral Convergence
            diff = mag_t - mag_p
            sc = diff.norm(p="fro") / (mag_t.norm(p="fro") + 1e-8)
            return sc
        else:  # magnitude L1 / L2
            if p == 1:
                return (mag_t - mag_p).abs().mean()
            elif p == 2:
                return F.mse_loss(mag_p, mag_t)
            else:
                raise ValueError("p must be 1 or 2")

    return _reg


# ── Early Stopping ────────────────────────────────────────────────────
class EarlyStopping:
    """Monitor validation loss and stop training if no improvement."""

    def __init__(self, patience=30, min_delta=0.0, verbose=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.best_loss  = float('inf')
        self.counter    = 0
        self._stop      = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            if self.verbose:
                print(f"  ↳ New best val loss: {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ↳ No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self._stop = True

    @property
    def should_stop(self):
        return self._stop
    
    
    
    
    