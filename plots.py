import matplotlib.pyplot as plt
import numpy as np

def make_smoothing_plot(t, raw, smoothed):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, raw, color='gray', alpha=0.5, label="Raw")
    if smoothed is not None:
        ax.plot(t, smoothed, 'b-', label="Smoothed")
    ax.set_title("Signal Smoothing")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return fig

def make_detection_plot(t, smoothed, peaks):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, smoothed, 'b-', label="Smoothed")
    if len(peaks) > 0:
        ax.scatter(t[peaks], smoothed[peaks], color='red', label="Peaks")
        ax.set_title(f"Peak Detection ({len(peaks)} peaks found)")
    else:
        ax.set_title("Peak Detection (No peaks found)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return fig

def make_analysis_plot(t, smoothed, start_time, end_time, selected_peak):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, smoothed, 'b-', alpha=0.5, label="Full Signal")
    if start_time is not None and end_time is not None:
        start_idx = int(np.searchsorted(t, start_time))
        end_idx = int(np.searchsorted(t, end_time))
        ax.plot(t[start_idx:end_idx], smoothed[start_idx:end_idx], 'r-', linewidth=2, label="Selected Segment")
    if selected_peak is not None and selected_peak < len(t):
        ax.axvline(x=t[selected_peak], color='green', linestyle='--', label="Peak Center")
    ax.set_title("Isolated Peak Analysis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.legend()
    return fig
