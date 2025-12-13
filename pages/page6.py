import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from scipy.fft import fft
import os
import pickle
from deps import handler, updater

st.set_page_config(page_title="FFT Analysis", layout="wide")
st.title("Frequency Domain Analysis (RR-Interval PSD)")
st.markdown("Power Spectral Density (PSD)")

st.sidebar.header("FFT Analysis Controls")
win_size = st.sidebar.slider("Window Size", 50, 500, 256, 1)
overlap_percent = st.sidebar.slider("Overlap Percentage", 0, 90, 50, 5)
overlap_val = int(win_size * (overlap_percent / 100))
st.sidebar.info(f"Current Overlap: {overlap_val} samples")
n_fft = st.sidebar.selectbox(
    "FFT Resolution (N-points)",
    options=[256, 512, 1024, 2048, 4096],
    index=3,
    help="Higher values increase frequency resolution (smoother plot)."
)
st.subheader("1. Load Data Files")
col1, col2 = st.columns(2)
tacho_filename = col1.text_input("Enter Tachogram Data Filename:", "rr_save")
raw_filename = col2.text_input("Enter Raw Data Filename:", "daffa_data")

if st.button("Load and Analyze Frequency Domain"):
    try:
        ppg = handler.load(f"temp/{raw_filename}")
        
        tacho_data = handler.load(f"temp/{tacho_filename}")

        if ppg is None or tacho_data is None:
            st.stop()
        fs_ppg = 50.0
        time_vector = np.array(ppg.time) / fs_ppg
        signal_values = np.array(ppg.value)

        rr_intervals_sec = np.array(tacho_data.time)

        if len(rr_intervals_sec) > 0:
            mean_rr_interval = np.mean(rr_intervals_sec)
            fs_rr = 1.0 / mean_rr_interval
        else:
            st.error("RR Interval data is empty. Cannot perform FFT analysis.")
            st.stop()

        st.subheader("Raw PPG Signal Preview")
        fig_raw = go.Figure(data=go.Scatter(x=time_vector, y=signal_values, mode='lines'))
        fig_raw.update_layout(title="Loaded Raw PPG Data", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_raw, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load necessary data. Error: {e}")
        st.stop()
    @st.cache_data
    def run_full_fft_analysis(rr_intervals, fs_rr, window_size, overlap, n_fft):
        def hamming(N):
            return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

        def calculate_psd(full_signal, fs, window_size, overlap, pad_to):
            step = window_size - overlap
            segments = []
            for start in range(0, len(full_signal) - window_size + 1, step):
                segment = np.array(full_signal[start:start + window_size])
                segment = segment - np.mean(segment)
                window = hamming(window_size)
                windowed_segment = segment * window
                padded_segment = np.pad(windowed_segment, (0, pad_to - window_size), 'constant')
                fft_result = fft(padded_segment)
                psd = (2 * (np.abs(fft_result) ** 2)) / (fs * np.sum(window**2))
                segments.append(psd[:pad_to // 2 + 1])
            psd_avg = np.mean(segments, axis=0)
            freqs = np.linspace(0, fs / 2, len(psd_avg))
            return freqs, psd_avg

        def band_power(freqs, psd, f_low, f_high):
            indices = np.where((freqs >= f_low) & (freqs < f_high))
            return np.trapz(psd[indices], freqs[indices])

        freqs, psd = calculate_psd(rr_intervals, fs_rr, window_size, overlap, pad_to=n_fft)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        lf_power_s2 = band_power(freqs, psd, *lf_band)
        hf_power_s2 = band_power(freqs, psd, *hf_band)
        total_power_s2 = lf_power_s2 + hf_power_s2
        scaling_factor = 10**2
        lf_power = lf_power_s2 * scaling_factor
        hf_power = hf_power_s2 * scaling_factor
        total_power = total_power_s2 * scaling_factor
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf
        lf_indices = np.where((freqs >= lf_band[0]) & (freqs < lf_band[1]))
        hf_indices = np.where((freqs >= hf_band[0]) & (freqs < hf_band[1]))
        peak_lf_freq = freqs[lf_indices][np.argmax(psd[lf_indices])] if len(lf_indices[0]) > 0 else 0
        peak_hf_freq = freqs[hf_indices][np.argmax(psd[hf_indices])] if len(hf_indices[0]) > 0 else 0
        features = {
            'TP': total_power, 'TP_of_LF': lf_power, 'TP_of_HF': hf_power,
            'LF_HF_Ratio': lf_hf_ratio,
            'LF_nu': (lf_power / total_power) * 100 if total_power > 0 else 0,
            'HF_nu': (hf_power / total_power) * 100 if total_power > 0 else 0,
            'Peak_Freq_LF': peak_lf_freq, 'Peak_Freq_HF': peak_hf_freq,
        }
        
        
        
        return freqs, psd * scaling_factor, features

    @st.cache_data
    def calculate_sd1(rr_intervals):
        rr_ms = rr_intervals 
        diffs = np.diff(rr_ms)
        return np.std(diffs) / np.sqrt(2)

    freqs, psd_ms2, features = run_full_fft_analysis(rr_intervals_sec, fs_rr, win_size, overlap_val, n_fft)
    sd1_metric = calculate_sd1(rr_intervals_sec)

    st.markdown("---")
    st.subheader("Summary of Calculated Features")
    summary_data = {
        "TP (ms²/Hz)": features['TP'], "TP of LF (ms²/Hz)": features['TP_of_LF'],
        "TP of HF (ms²/Hz)": features['TP_of_HF'], "LF/HF Ratio": features['LF_HF_Ratio'],
        "LF (n.u.)": features['LF_nu'], "HF (n.u.)": features['HF_nu'],
        "Peak Frequency of LF (Hz)": features['Peak_Freq_LF'],
        "Peak Frequency of HF (Hz)": features['Peak_Freq_HF']
    }
    summary_df = pd.DataFrame.from_dict(summary_data, orient='index', columns=['Value'])
    summary_df.index.name = "Metric"
    st.dataframe(summary_df.style.format("{:.4f}"))

    st.markdown("---")
    plot_col1, plot_col2 = st.columns([2, 1])
    with plot_col1:
        st.subheader("Power Spectral Density (PSD)")
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs, y=psd_ms2, mode='lines', name='PSD', line=dict(color='blue')))
        fig_psd.add_vrect(x0=0.003, x1=0.04, fillcolor="green", opacity=0.2, layer="below", line_width=0, name='VLF')
        fig_psd.add_vrect(x0=0.04, x1=0.15, fillcolor="yellow", opacity=0.2, layer="below", line_width=0, name='LF')
        fig_psd.add_vrect(x0=0.15, x1=0.4, fillcolor="red", opacity=0.2, layer="below", line_width=0, name='HF')
        fig_psd.update_layout(
            title='PSD with Frequency Bands', xaxis_title='Frequency (Hz)',
            yaxis_title='Power Spectral Density (ms²/Hz)',
            xaxis=dict(range=[0, 0.7], autorange=False)
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    with plot_col2:
        st.subheader("Normalized Power Distribution")
        fig_bar = go.Figure(data=[go.Bar(
            x=['LF (n.u.)', 'HF (n.u.)'],
            y=[features['LF_nu'], features['HF_nu']],
            text=[f"{features['LF_nu']:.1f}%", f"{features['HF_nu']:.1f}%"],
            textposition='auto', marker_color=['blue', 'red']
        )])
        fig_bar.update_layout(
            title=f"LF/HF Ratio: {features['LF_HF_Ratio']:.2f}",
            yaxis_title="Normalized Units (%)", yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_bar, use_container_width=True)