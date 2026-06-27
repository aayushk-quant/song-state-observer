import librosa
import numpy as np

def load_audio(path, sr = 22050, mono = True):
    requested_sr = sr
    y, actual_sr = librosa.load(path, sr = requested_sr, mono = mono)

    if y.ndim != 1:
        raise ValueError("Expected a 1-D waveform. Use mono = True.")
    
    if not np.all(np.isfinite(y)):
        raise ValueError("Loaded audio contains NaNs or infinite values.")

    return y, actual_sr

def _fine_features(y, sr, n_fft = 2048, hop_length = 512):
    y = np.asarray(y, dtype = float)
    if y.ndim != 1:
        raise ValueError("Expected a 1-D waveform.")
    
    S = np.abs(librosa.stft(y, n_fft = n_fft, hop_length = hop_length))

    rms = librosa.feature.rms(S = S, frame_length = n_fft, hop_length = hop_length)
    centroid = librosa.feature.spectral_centroid(S = S, sr = sr, n_fft = n_fft, hop_length = hop_length)
    flatness = librosa.feature.spectral_flatness(S = S, n_fft = n_fft, hop_length = hop_length)

    rms = rms.squeeze(0)
    centroid = centroid.squeeze(0)
    flatness = flatness.squeeze(0)

    centroid = np.nan_to_num(centroid, nan = 0.0)
    flatness = np.nan_to_num(flatness, nan = 0.0)

    return {"rms": rms, "centroid": centroid, "flatness": flatness}

def _aggregate_to_windows(values, sr, hop_length, win = 1.0, hop = 0.5):
    fine_times = np.arange(len(values)) * hop_length / sr
    duration = len(values) * hop_length / sr
    if duration < win:
        raise ValueError("duration cannot be less than win")

    T = int(np.floor((duration - win) / hop)) + 1
    if T <= 0:
        raise ValueError("T must be positive")
    
    out = np.empty(T)
    for w in range(T):
        start = w * hop
        end = start + win
        mask = (fine_times >= start) & (fine_times < end)
        out[w] = values[mask].mean()
    
    return out

def extract_features(path, sr = 22050, n_fft = 2048, hop_length = 512, win = 1.0, hop = 0.5, eps = 1e-8):
    y, sr = load_audio(path)
    f = _fine_features(y, sr)
    aggregated = {
        name: _aggregate_to_windows(series, sr=sr, hop_length=hop_length, win=win, hop=hop)
        for name, series in f.items()
    }
    log_rms = np.log(aggregated["rms"] + eps)
    centroid = aggregated["centroid"]
    flatness = aggregated["flatness"]
    delta_log_rms = np.diff(log_rms, prepend = log_rms[0])
    delta_flatness = np.diff(flatness, prepend = flatness[0])

    features = np.column_stack([log_rms, centroid, flatness, delta_log_rms, delta_flatness])
    return features