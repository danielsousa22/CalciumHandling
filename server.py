from shiny import reactive, render, ui
import pandas as pd
import numpy as np
from analysis_utils import smooth_signal, detect_peaks_with_skip, compute_metrics, estimate_bpm
from plot_utils import make_smoothing_plot, make_detection_plot, make_analysis_plot

def server(input, output, session):
    raw = reactive.Value(None)
    smooth = reactive.Value(None)
    step = reactive.Value(1)
    peaks = reactive.Value([])
    metrics = reactive.Value({})

    @reactive.Effect
    @reactive.event(input.file)
    def load():
        df = pd.read_csv(input.file()[0]['datapath'])
        df.columns = ['frame','intensity'] if df.shape[1]==2 else df.columns
        raw.set(df)
        smooth.set(None)
        step.set(1)

    @reactive.Effect
    @reactive.event(input.update)
    def do_smooth():
        df = raw.get()
        w = input.smooth_window(); p = input.smooth_poly()
        smooth.set(smooth_signal(df['intensity'], w, p))

    @reactive.Calc
    def det_peaks():
        return detect_peaks_with_skip(smooth.get() or [], input.height(), input.distance(), input.skip_first(), input.skip_last())

    @reactive.Effect
    @reactive.event(input.next1)
    def to2(): step.set(2)
    @reactive.Effect
    @reactive.event(input.back2)
    def b1(): step.set(1)
    @reactive.Effect
    @reactive.event(input.next2)
    def to3():
        p = det_peaks()
        peaks.set(p)
        step.set(3)

    @reactive.Effect
    @reactive.event(input.analyze)
    def analyze():
        df = raw.get(); t = df['frame']/input.frame_rate()
        p = peaks.get()
        bpm = estimate_bpm(p, t.values)
        # smallest segment around selected peak
        idx = p[int(input.peak_sel())]
        seg_idx = np.searchsorted(t, [input.start(), input.end()])
        x = smooth.get()[seg_idx[0]:seg_idx[1]]
        tt = t.values[seg_idx[0]:seg_idx[1]]
        m = compute_metrics(x, tt, np.argmax(x))
        metrics.set({'bpm':bpm, 'amplitude':m['amplitude'], 'time_to_peak':m['time_to_peak']})

    @render.plot
    def plot1():
        df = raw.get(); t = df['frame']/input.frame_rate()
        return make_smoothing_plot(t.values, df['intensity'].values, smooth.get())
    @render.plot
    def detect_plot():
        df = raw.get(); t = df['frame']/input.frame_rate()
        return make_detection_plot(t.values, smooth.get() or [], det_peaks())
    @render.plot
    def anal_plot():
        df = raw.get(); t = df['frame']/input.frame_rate()
        return make_analysis_plot(t.values, smooth.get() or [], input.start(), input.end(), int(input.peak_sel()))

    @render.ui
    def metrics2():
        p=det_peaks(); df=raw.get(); t=df['frame']/input.frame_rate()
        return ui.div(f"Detected peaks: {len(p)}")

    @render.ui
    def final_metrics():
        m=metrics.get()
        rows = [ui.tr(ui.th('Metric'), ui.th('Value'))] + [
            ui.tr(ui.td(k), ui.td(f"{v:.2f}")) for k,v in m.items()
        ]
        return ui.div(ui.h4('Results'), ui.table(*rows))

    @reactive.Effect
    def flip():
        js = f"$('.cond').removeClass('active');$('#step{step.get()}').addClass('active');"
        ui.insert_ui(ui.tags.script(js), selector='body', where='afterEnd')