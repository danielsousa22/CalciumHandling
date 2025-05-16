
import os
import argparse
import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration & Logging
# --------------------------------------------------
def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# --------------------------------------------------
# Signal Processing Utilities
# --------------------------------------------------
def smooth_signal(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    return savgol_filter(signal, window_length=window, polyorder=polyorder)

# --------------------------------------------------
# Decay Model
# --------------------------------------------------
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

# --------------------------------------------------
# Peak & Local Detection Utilities
# --------------------------------------------------
def detect_peaks_with_skip(signal: np.ndarray, height: float, distance: int,
                           skip_first: bool=False, skip_last: bool=False) -> tuple:
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    if skip_first and len(peaks) > 0:
        peaks = peaks[1:]
        for k in properties:
            properties[k] = properties[k][1:]
    if skip_last and len(peaks) > 0:
        peaks = peaks[:-1]
        for k in properties:
            properties[k] = properties[k][:-1]
    return peaks, properties

def detect_first_local_peak(signal: np.ndarray, peak_indices: np.ndarray,
                            avg_bpm: float, frame_rate: float,
                            before_sec: float=0, after_sec: float=0) -> dict:
    if avg_bpm <= 0:
        raise ValueError("Average BPM must be positive to define window size.")
    cycle_samples = int((60 / avg_bpm) * frame_rate)
    if len(peak_indices) == 0:
        raise ValueError("No peaks available to select from.")
    first_idx = peak_indices[0]
    start = max(first_idx - cycle_samples//2 - int(before_sec * frame_rate), 0)
    end = min(first_idx + cycle_samples//2 + int(after_sec * frame_rate), len(signal)-1)
    window = signal[start:end]
    local_peaks, _ = find_peaks(window)
    if len(local_peaks) == 0:
        raise ValueError("No local peaks in window.")
    selected = local_peaks[0] + start
    return {'window_range': (start, end), 'selected_peak': selected}

# --------------------------------------------------
# Analysis Metrics
# --------------------------------------------------
def calculate_bpm(peak_indices: np.ndarray, frame_rate: float) -> tuple:
    times = peak_indices / frame_rate
    intervals = np.diff(times)
    bpm = 60 / intervals
    return bpm, np.mean(bpm) if len(bpm) else 0

def compute_baseline(signal: np.ndarray, percentile: float=10) -> float:
    return np.percentile(signal, percentile)

def compute_amplitude(signal: np.ndarray, baseline: float, peak_idx: int) -> float:
    return signal[peak_idx] - baseline

def compute_time_to_peak(time: np.ndarray, peak_idx: int) -> float:
    return time[peak_idx] - time[0]

def fit_signal_decay(time: np.ndarray, signal: np.ndarray, peak_idx: int) -> dict:
    t_decay = time[peak_idx:] - time[peak_idx]
    y_decay = signal[peak_idx:]
    baseline = compute_baseline(signal)
    A0 = signal[peak_idx] - baseline
    p0 = (A0, np.max(t_decay)/2, baseline)
    popt, _ = curve_fit(exp_decay, t_decay, y_decay, p0=p0)
    A_fit, tau, C_fit = popt
    return {'tau': tau, 'A_fit': A_fit, 'C': C_fit,
            't_decay': t_decay, 'y_fit': exp_decay(t_decay, *popt)}

def compute_decay_times(time: np.ndarray, signal: np.ndarray, peak_idx: int,
                        tau: float, levels: list=[0.8, 0.5, 0.1]) -> dict:
    baseline = compute_baseline(signal)
    amp = signal[peak_idx] - baseline
    target_vals = {f: baseline + amp*f for f in levels}
    t_decay = time[peak_idx:]
    s_decay = signal[peak_idx:]
    results = {}
    for frac, val in target_vals.items():
        idx = np.where(s_decay <= val)[0]
        results[frac] = (t_decay[idx[0]] - time[peak_idx]) if idx.size else np.nan
    return results

# --------------------------------------------------
# Plotting Utilities
# --------------------------------------------------
def plot_peaks(time: np.ndarray, signal: np.ndarray, peaks: np.ndarray, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, label='Signal')
    plt.scatter(time[peaks], signal[peaks], c='red', label='Peaks')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
    plt.close()

def plot_window(signal: np.ndarray, window_range: tuple, selected_peak: int,
                frame_rate: float, title: str):
    start, end = window_range
    t = np.arange(start, end) / frame_rate
    plt.figure(figsize=(8, 3))
    plt.plot(t, signal[start:end], label='Window')
    plt.axvline(selected_peak/frame_rate, linestyle='--', label='Peak')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
    plt.close()

# --------------------------------------------------
# Interactive File Processing
# --------------------------------------------------
def process_file_interactive(filepath: str, frame_rate: float,
                             smooth_params: dict, output_folder: str):
    df = pd.read_csv(filepath)
    signal = smooth_signal(df['intensity'].values, **smooth_params)
    time = df['frame'].values / frame_rate

    # Interactive peak detection
    while True:
        height = float(input('Enter peak height threshold: '))
        distance = int(input('Enter minimum peak distance (samples): '))
        skip_first = input('Skip first peak? (y/n): ').lower() == 'y'
        skip_last = input('Skip last peak? (y/n): ').lower() == 'y'
        peaks, _ = detect_peaks_with_skip(signal, height, distance,
                                          skip_first, skip_last)
        plot_peaks(time, signal, peaks, title=os.path.basename(filepath))
        if input('Satisfied with peak detection? (y/n): ').lower() == 'y':
            break

    bpm_vals, avg_bpm = calculate_bpm(peaks, frame_rate)
    print(f'Average BPM: {avg_bpm:.2f}')

    # Interactive fine-tuning
    while True:
        if input('Perform local window fine-tuning? (y/n): ').lower() != 'y':
            break
        before_sec = float(input('Seconds before peak: '))
        after_sec = float(input('Seconds after peak: '))
        info = detect_first_local_peak(signal, peaks, avg_bpm,
                                       frame_rate, before_sec, after_sec)
        plot_window(signal, info['window_range'],
                    info['selected_peak'], frame_rate,
                    title='Local Window')
        if input('Satisfied with fine-tuning? (y/n): ').lower() == 'y':
            break

    # Compute Metrics
    baseline = compute_baseline(signal)
    selected_peak = info['selected_peak'] if 'info' in locals() else peaks[0]
    amplitude = compute_amplitude(signal, baseline, selected_peak)
    t_peak = compute_time_to_peak(time, selected_peak)
    decay = fit_signal_decay(time, signal, selected_peak)
    decay_times = compute_decay_times(time, signal, selected_peak, decay['tau'])

    # Display metrics
    print(f"Baseline (10th pct): {baseline:.3f}")
    print(f"Amplitude: {amplitude:.3f}")
    print(f"Time to peak: {t_peak:.3f} s")
    print(f"Tau (decay constant): {decay['tau']:.3f} s")
    for frac, dt in decay_times.items():
        print(f"Time to {int(frac*100)}% decay: {dt:.3f} s")

    # Save Summary
    name = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(output_folder, exist_ok=True)
    summary = pd.DataFrame({
        'metric': ['baseline','amplitude','time_to_peak','tau'] +
                  [f'decay_{int(f*100)}' for f in decay_times],
        'value': [baseline, amplitude, t_peak, decay['tau']] +
                  list(decay_times.values())
    })
    summary.to_csv(os.path.join(output_folder,
                                f"{name}_metrics.csv"), index=False)
    print(f"Metrics saved to {name}_metrics.csv")

# --------------------------------------------------
# Main Entry Point
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive peak analysis")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file or folder')
    parser.add_argument('-o', '--output', default='./results', help='Output folder')
    parser.add_argument('--frame_rate', type=float, default=500, help='Frame rate')
    parser.add_argument('--smooth_window', type=int, default=15, help='Smoothing window length')
    parser.add_argument('--smooth_poly', type=int, default=3, help='Smoothing polyorder')
    parser.add_argument('--log', default='INFO', help='Logging level')
    args = parser.parse_args()

    setup_logging(args.log)
    smooth_params = {'window': args.smooth_window, 'polyorder': args.smooth_poly}

    paths = [args.input]
    if os.path.isdir(args.input):
        paths = [os.path.join(args.input, f)
                 for f in os.listdir(args.input) if f.endswith('.csv')]

    for p in paths:
        logging.info(f"Processing {p}")
        process_file_interactive(p, args.frame_rate, smooth_params, args.output)

if __name__ == '__main__':
    main()
