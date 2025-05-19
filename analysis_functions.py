import numpy as np
from scipy.signal import savgol_filter, find_peaks

def smooth_signal(x, window, polyorder):
    window = min(window, max(1, (len(x) // 2) * 2 + 1))
    return savgol_filter(x, window_length=window, polyorder=polyorder)

def detect_peaks_with_skip(x, height, distance, skip_first, skip_last):
    if len(x) == 0:
        return np.array([])
    peaks, _ = find_peaks(x, height=height, distance=distance)
    if skip_first and len(peaks) > 0:
        peaks = peaks[1:]
    if skip_last and len(peaks) > 0:
        peaks = peaks[:-1]
    return peaks

def compute_metrics(x, t, peak_idx):
    try:
        baseline = np.percentile(x, 10)
        amplitude = x[peak_idx] - baseline
        time_to_peak = t[peak_idx] - t[0]

        # Estimate decay times based on thresholds of amplitude
        results = {}
        for frac in [0.8, 0.5, 0.1]:
            tgt = baseline + amplitude * frac
            idx = np.where(x[peak_idx:] <= tgt)[0]
            results[f"decay_{int(frac * 100)}"] = t[peak_idx + idx[0]] - t[peak_idx] if len(idx) > 0 else np.nan

        return {
            "baseline": baseline,
            "amplitude": amplitude,
            "time_to_peak": time_to_peak,
            **results
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            "baseline": np.nan,
            "amplitude": np.nan,
            "time_to_peak": np.nan,
            "decay_80": np.nan,
            "decay_50": np.nan,
            "decay_10": np.nan
        }

def estimate_bpm(peaks, t):
    if len(peaks) < 2:
        return np.nan
    intervals = np.diff(t[peaks])
    mean_interval = np.mean(intervals)
    bpm = 60.0 / mean_interval if mean_interval > 0 else np.nan
    return bpm