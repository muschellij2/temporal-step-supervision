# Optional handcrafted signal features for ablation or classical baselines.

import numpy as np
import scipy.signal as sps


def zero_crossing_rate(x):
    """Return the fraction of adjacent samples that cross zero."""
    return ((x[:-1] * x[1:]) < 0).mean() if len(x) > 1 else 0.0


def spectral_centroid(x, fs):
    """Compute the power-weighted average frequency."""
    f, Pxx = sps.welch(x, fs=fs, nperseg=min(len(x), 512))
    return (f * Pxx).sum() / Pxx.sum() if Pxx.sum() > 0 else 0.0


def bandpower(x, fs, fmin=0.5, fmax=3.5):
    """Integrate spectral power over the target walking-frequency band."""
    f, Pxx = sps.welch(x, fs=fs, nperseg=min(len(x), 512))
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[idx], f[idx]) if idx.any() else 0.0


def dominant_freq_and_power(x, fs):
    """Return the dominant frequency and its spectral power."""
    f, Pxx = sps.welch(x, fs=fs, nperseg=min(len(x), 512))
    if len(Pxx) == 0:
        return 0.0, 0.0
    i = np.argmax(Pxx)
    return f[i], Pxx[i]


def moving_average(x, k):
    """Smooth a sequence with a centered moving average."""
    if k <= 1:
        return x.copy()
    return np.convolve(x, np.ones(k) / k, mode="same")


def lag_mean_diff(x, lag):
    """Difference between means separated by a fixed lag."""
    if len(x) <= lag:
        return 0.0
    return float(np.mean(x[lag:]) - np.mean(x[:-lag]))


def compute_window_features(wv, fs):
    """Build a compact statistical and spectral feature vector for one window."""
    mean, std = np.mean(wv), np.std(wv) + 1e-8
    rms = np.sqrt(np.mean(wv**2))
    mad = np.mean(np.abs(wv - mean))
    skew = ((wv - mean)**3).mean() / (std**3)
    kurt = ((wv - mean)**4).mean() / (std**4) - 3
    zcr = zero_crossing_rate(wv)
    sc = spectral_centroid(wv, fs)
    bp = bandpower(wv, fs, fmin=0.5, fmax=3.5)
    domf, domp = dominant_freq_and_power(wv, fs)

    basic9 = [rms, mad, skew, kurt, zcr, sc, bp, domf, domp]

    ma3, ma5, ma10, ma15 = [np.mean(moving_average(wv, k)) for k in [3, 5, 10, 15]]
    lag5, lag10, lag15 = [lag_mean_diff(wv, k) for k in [5, 10, 15]]
    return np.array(basic9 + [ma3, ma5, ma10, ma15, lag5, lag10, lag15], dtype=np.float32)
