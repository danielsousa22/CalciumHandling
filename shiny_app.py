
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --------------------------------------------------
# Analysis Functions
# --------------------------------------------------
def smooth_signal(x, window, polyorder):
    window = min(window, max(1, (len(x) // 2) * 2 + 1))
    return savgol_filter(x, window_length=window, polyorder=polyorder)

def detect_peaks_with_skip(x, height, distance, skip_first, skip_last):
    peaks, _ = find_peaks(x, height=height, distance=distance)
    if skip_first and peaks.size:
        peaks = peaks[1:]
    if skip_last and peaks.size:
        peaks = peaks[:-1]
    return peaks

def detect_first_local_peak(x, peaks, avg_bpm, fs, before, after):
    cycle = int((60/avg_bpm)*fs) if avg_bpm>0 else 0
    if peaks.size == 0:
        return None
    idx0 = peaks[0]
    start = max(0, idx0 - cycle//2 - int(before*fs))
    end   = min(len(x), idx0 + cycle//2 + int(after*fs))
    window = x[start:end]
    local, _ = find_peaks(window)
    if local.size == 0:
        return None
    return start, end, local[0]+start

def compute_metrics(x, t, peak_idx):
    baseline = np.percentile(x, 10)
    amplitude = x[peak_idx] - baseline
    time_to_peak = t[peak_idx] - t[0]
    def exp_decay(t_, A, tau, C): return A*np.exp(-t_/tau)+C
    te = t - t[peak_idx]
    ye = x[peak_idx:]
    popt, _ = curve_fit(exp_decay, te, ye, p0=(amplitude, te.max()/2, baseline))
    tau = popt[1]
    results = {}
    for frac in [0.8,0.5,0.1]:
        tgt = baseline + amplitude*frac
        idx = np.where(ye<=tgt)[0]
        results[f"decay_{int(frac*100)}"] = te[idx[0]] if idx.size else np.nan
    return {
        "baseline": baseline,
        "amplitude": amplitude,
        "time_to_peak": time_to_peak,
        "tau": tau,
        **results
    }

# --------------------------------------------------
# UI
# --------------------------------------------------
app_ui = ui.page_fluid(
    ui.h2("Interactive Peak Analysis (Python Shiny)"),
    ui.input_file("file", "Upload CSV", accept=[".csv"]),
    ui.input_numeric("frame_rate", "Frame Rate (Hz)", value=500),
    ui.input_slider("smooth_window", "Smoothing Window", min=5, max=101, value=15, step=2),
    ui.input_numeric("smooth_poly", "Smoothing Polyorder", value=3, min=1),
    ui.h4("Peak Detection"),
    ui.input_numeric("height", "Height Threshold", value=100),
    ui.input_numeric("distance", "Min Peak Distance", value=300),
    ui.input_checkbox("skip_first", "Skip First Peak", False),
    ui.input_checkbox("skip_last", "Skip Last Peak", False),
    ui.input_action_button("detect", "Detect Peaks"),
    ui.output_plot("peak_plot"),
    ui.h4("Fine-Tuning"),
    ui.input_numeric("before_sec", "Seconds Before Peak", value=0),
    ui.input_numeric("after_sec", "Seconds After Peak", value=0),
    ui.input_action_button("tune", "Fine-Tune"),
    ui.output_plot("window_plot"),
    ui.h4("Metrics"),
    ui.input_action_button("compute", "Compute Metrics"),
    ui.output_table("metrics_table")
)

# --------------------------------------------------
# Server
# --------------------------------------------------
def server(input, output, session):
    df        = reactive.Value(None)
    t_vec     = reactive.Value(None)
    signal    = reactive.Value(None)
    peaks     = reactive.Value(None)
    local_inf = reactive.Value(None)
    metrics   = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.file)
    def _load():
        f = input.file()
        if not f:
            return
        data = pd.read_csv(f[0]["datapath"])
        if data.shape[1] == 2:
            data.columns = ["frame", "intensity"]
        elif not {"frame", "intensity"}.issubset(data.columns):
            session.show_notification(
                "CSV must have exactly 2 columns or be named 'frame' and 'intensity'",
                type="error"
            )
            return

        df.set(data)
        fs = input.frame_rate()
        t = data["frame"].values / fs
        x = smooth_signal(data["intensity"].values, input.smooth_window(), input.smooth_poly())
        t_vec.set(t)
        signal.set(x)

    @reactive.Effect
    @reactive.event(input.detect)
    def _detect():
        x = signal.get()
        if x is None:
            return
        p = detect_peaks_with_skip(
            x, input.height(), input.distance(),
            input.skip_first(), input.skip_last()
        )
        peaks.set(p)

    @output
    @render.plot
    def peak_plot():
        p = peaks.get()
        x = signal.get()
        t = t_vec.get()
        if x is None or p is None:
            return None
        fig, ax = plt.subplots()
        ax.plot(t, x, label="Signal")
        ax.scatter(t[p], x[p], c="red", label="Peaks")
        ax.set_title("Detected Peaks")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity")
        ax.legend()
        plt.tight_layout()
        return fig

    @reactive.Effect
    @reactive.event(input.tune)
    def _tune():
        p = peaks.get(); x = signal.get(); t = t_vec.get()
        if x is None or p is None:
            return
        intervals = np.diff(p / input.frame_rate())
        avg_bpm = np.mean(60/intervals) if intervals.size else 0
        info = detect_first_local_peak(
            x, p, avg_bpm,
            input.frame_rate(), input.before_sec(), input.after_sec()
        )
        local_inf.set(info)

    @output
    @render.plot
    def window_plot():
        info = local_inf.get()
        x = signal.get()
        t = t_vec.get()
        if info is None or x is None:
            return None
        st, en, sel = info
        fig, ax = plt.subplots()
        ax.plot(t[st:en], x[st:en], label="Window")
        ax.axvline(t[sel], color="green", linestyle="--", label="Selected Peak")
        ax.set_title("Local Window")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity")
        ax.legend()
        plt.tight_layout()
        return fig

    @reactive.Effect
    @reactive.event(input.compute)
    def _compute():
        info = local_inf.get()
        x    = signal.get()
        t    = t_vec.get()
        if info is None or x is None:
            return
        _, _, sel = info
        m = compute_metrics(x, t, sel)
        metrics.set(m)

    @output
    @render.table
    def metrics_table():
        m = metrics.get()
        if m is None:
            return None
        return pd.DataFrame.from_dict(m, orient="index", columns=["value"])

app = App(app_ui, server)

