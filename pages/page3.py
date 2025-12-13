import streamlit as st
from dwt_coeff import DWTCoeff
from deps import handler
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

try:
    available_files = [f for f in os.listdir('./temp/') if f.endswith('.dat')]
except FileNotFoundError:
    available_files = []

selected_file = st.selectbox(
    "Select the data session to analyze:",
    options=available_files
)
selected_file = selected_file.replace('.dat', '')

st.markdown("---")
var = handler.load(f"temp/{selected_file}")
# const
fs = 125
dwt6 = var.value
timedwt6 = var.time / 50
total = len(dwt6)

def moving_average_filter(signal, M):
    y1 = np.zeros_like(signal, dtype=float)
    for i in range(len(signal)):
        for j in range(M):
            if i - j >= 0:
                y1[i] += signal[i - j]
        y1[i] /= M

    y2 = np.zeros_like(y1, dtype=float)
    for i in range(len(y1)):
        for j in range(M):
            if i + j < len(y1):
                y2[i] += y1[i + j]
        y2[i] /= M

    return y2

def deteksi_puncak(sinyal, persen):
    maks = max(abs(x) for x in sinyal)  
    batas = persen * maks
    indeks = []
    nilai = []

    for i in range(1, len(sinyal)-1):
        if sinyal[i] > sinyal[i-1] and sinyal[i] > sinyal[i+1]:
            if sinyal[i] > batas:
                indeks.append(i)
                nilai.append(sinyal[i])
    return indeks, nilai

def buat_tabel_rr_rrinterval(waktu, indeks):
    waktu = waktu/50

    if len(indeks) < 2:
        return pd.DataFrame(), [], []
        
    interval_detik = [float((waktu[indeks[i+1]] - waktu[indeks[i]])) for i in range(len(indeks)-1)]
    bpm_per_beat = [60/i for i in interval_detik if i > 0]
    beat_labels = [f"{(waktu[indeks[i]]):.2f} s to {(waktu[indeks[i+1]]):.2f} s" for i in range(len(indeks)-1)]

    df = pd.DataFrame({
        "Beat Interval (waktu)": beat_labels,
        "RR Interval (s)": interval_detik,
        "Respiratory Rate (BPM)": bpm_per_beat
    })

    df["RR Interval (s)"] = df["RR Interval (s)"].round(3)
    df["Respiratory Rate (BPM)"] = df["Respiratory Rate (BPM)"].round(2)
    return df, interval_detik, bpm_per_beat
st.markdown("### 1. Signal Filtering")
M = 60 
st.write(f"Filter window size **M = {M}**")
filtered_dwt6 = moving_average_filter(dwt6, M)

fig_filter = go.Figure()
fig_filter.add_trace(go.Scatter(x=timedwt6, y=dwt6, mode='lines', name='Original dwt6 Signal', line=dict(color='rgba(23, 190, 207, 0.6)')))
fig_filter.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name=f'Filtered Signal (M={M})', line=dict(color='rgba(214, 39, 40, 1.0)', width=2)))
fig_filter.update_layout(title_text='Comparison of Original and Moving Average Filtered Data', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_filter, use_container_width=True)

st.markdown("### 2. Peak Detection and Respiratory Rate Analysis")
st.write("This section uses the **filtered signal** to detect peaks and calculate the respiratory rate.")

peak_threshold_percent = 0
idx_peaks, val_peaks = deteksi_puncak(filtered_dwt6, peak_threshold_percent)


fig_peaks = go.Figure()
fig_peaks.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name='Filtered Signal', line=dict(color='blue')))
fig_peaks.add_trace(go.Scatter(x=[timedwt6[i] for i in idx_peaks], y=val_peaks, mode='markers', name='Detected Peaks', marker=dict(color='red', symbol='x', size=8)))
fig_peaks.update_layout(title='Detected Peaks on Filtered Signal', xaxis_title='Time (s)', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_peaks, use_container_width=True)

st.write(f"**Total peaks detected:** {len(idx_peaks)}")

df_rr, rr_intervals, bpm_values = buat_tabel_rr_rrinterval(timedwt6, idx_peaks)
st.write(df_rr)
print(len(df_rr))

save_filename = st.text_input("Name for RR Calculation Result:")

if st.button("Save BR Calculations", key="LOADDATAKEY"):
    handler.save(df_rr['RR Interval (s)']/50, df_rr['Respiratory Rate (BPM)'], f"temp/{save_filename}")

# st.table(rr_intervals )

