import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from scipy.stats import entropy
from scipy.signal import welch
from scipy.spatial.distance import cosine, pdist, squareform
from ripser import ripser
from persim import wasserstein
import matplotlib.pyplot as plt


# ========================= Basic Utility Functions =========================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (sMAPE)."""
    return 1 - np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))


# ========================= Fractal Dimension =========================
def correlation_integral(data, r):
    """Compute the correlation integral for radius r."""
    dist_matrix = squareform(pdist(data))
    N = len(data)
    return np.sum(dist_matrix < r) / (N * (N - 1))


def correlation_dimension(data, r_vals):
    """Compute the correlation integral curve over a range of radii."""
    C_r_vals = [correlation_integral(data, r) for r in r_vals]
    return r_vals, C_r_vals


def compute_fractal_dimension(data, r_vals):
    """
    Estimate the fractal dimension by fitting a slope
    on the log-log plot of correlation integral vs radius.
    """
    r_vals, C_r_vals = correlation_dimension(data, r_vals)
    log_r = np.log(r_vals)
    log_C_r = np.log(C_r_vals)
    slope, _ = np.polyfit(log_r, log_C_r, 1)
    return slope


def compare_correlation_dimensions(orig_data, recon_data):
    """
    Compare the fractal dimensions of two systems
    and measure their similarity.
    """
    dist = squareform(pdist(orig_data))
    min_r = np.min(dist[np.nonzero(dist)])
    max_r = np.max(dist)
    r_vals = np.logspace(np.log10(min_r), np.log10(max_r), 50)

    fd_orig = compute_fractal_dimension(orig_data, r_vals)
    fd_recon = compute_fractal_dimension(recon_data, r_vals)
    sim = smape(fd_orig, fd_recon)

    return fd_orig, fd_recon, sim


# ========================= Distribution Similarity =========================
def calculate_kl_divergence_multi_channel(ts1, ts2, num_bins='auto'):
    """Compute KL divergence for each channel of two time series."""
    n_channels = ts1.shape[1]
    kl_div = np.zeros(n_channels)
    for ch in range(n_channels):
        s1, s2 = ts1[:, ch], ts2[:, ch]
        if np.any(np.isnan(s1)) or np.any(np.isnan(s2)):
            kl_div[ch] = np.nan
            continue
        bins = np.histogram_bin_edges(np.hstack((s1, s2)), bins=num_bins)
        p, _ = np.histogram(s1, bins=bins, density=True)
        q, _ = np.histogram(s2, bins=bins, density=True)
        p += 1e-10; q += 1e-10  # numerical stability
        p /= p.sum(); q /= q.sum()
        kl_div[ch] = entropy(p, q)
    return kl_div


def calculate_hellinger_distance_multi_channel(ts1, ts2, num_bins='auto'):
    """Compute Hellinger distance for each channel of two time series."""
    n_channels = ts1.shape[1]
    h_dist = np.zeros(n_channels)
    for ch in range(n_channels):
        s1, s2 = ts1[:, ch], ts2[:, ch]
        if np.any(np.isnan(s1)) or np.any(np.isnan(s2)):
            h_dist[ch] = np.nan
            continue
        bins = np.histogram_bin_edges(np.hstack((s1, s2)), bins=num_bins)
        p, _ = np.histogram(s1, bins=bins, density=True)
        q, _ = np.histogram(s2, bins=bins, density=True)
        p += 1e-10; q += 1e-10  # numerical stability
        p /= p.sum(); q /= q.sum()
        h = (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
        h_dist[ch] = h
    return h_dist


# ========================= Structural & Spectral Similarity =========================
def functional_connectivity_similarity(ts1, ts2):
    """
    Compute structural similarity between two systems
    based on their correlation matrices.
    """
    corr1, corr2 = np.corrcoef(ts1), np.corrcoef(ts2)
    diff = np.linalg.norm(corr1 - corr2, ord='fro')
    max_norm = np.linalg.norm(corr1, ord='fro') + np.linalg.norm(corr2, ord='fro')
    return 1 - (diff / max_norm)


def compare_spectrum(ts1, ts2, fs=200):
    """
    Compare spectral similarity (per channel) between two time series
    using Welch’s method and cosine similarity.
    """
    f1, Pxx1 = welch(ts1, fs=fs)
    f2, Pxx2 = welch(ts2, fs=fs)
    Pxx1 /= np.sum(Pxx1)
    Pxx2 /= np.sum(Pxx2)
    return 1 - cosine(Pxx1, Pxx2)


# ========================= Main Evaluation Function =========================
def regression_metrics(y_true, y_pred):
    """
    Evaluate regression performance and additional dynamical system metrics.

    Args:
        y_true, y_pred: arrays of shape (n_channels, T)

    Returns:
        DataFrame with regression and similarity metrics per channel,
        and additional attributes for topological and connectivity similarity.
    """
    dgm_true = ripser(y_true.T, maxdim=1)['dgms'][1]
    dgm_pred = ripser(y_pred.T, maxdim=1)['dgms'][1]

    fd_true, fd_pred, fd_sim = compare_correlation_dimensions(y_true.T, y_pred.T)
    kl = calculate_kl_divergence_multi_channel(y_true.T, y_pred.T)
    hell = calculate_hellinger_distance_multi_channel(y_true.T, y_pred.T)
    fc_sim = functional_connectivity_similarity(y_true.T, y_pred.T)
    topo_dist = wasserstein(dgm_true, dgm_pred)

    rows = []
    for ch in range(y_true.shape[0]):
        yt, yp = y_true[ch, :], y_pred[ch, :]
        rows.append(dict(
            ch = ch,
            MSE = mean_squared_error(yt, yp),
            MAE = mean_absolute_error(yt, yp),
            EV  = explained_variance_score(yt, yp),
            R2  = r2_score(yt, yp),
            spec_sim = compare_spectrum(yt, yp),
            corr_sim = fd_sim,
            KL = kl[ch],
            hellinger = hell[ch]
        ))
    df = pd.DataFrame(rows).set_index('ch')
    df.loc['mean'] = df.mean()
    df.attrs['fc_sim'] = fc_sim
    df.attrs['h1_wass_dist'] = topo_dist
    return df
