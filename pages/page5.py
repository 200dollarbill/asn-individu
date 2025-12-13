import streamlit as st
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
from deps import handler
st.set_page_config(layout="wide")
st.header("Section 2: Heart Rate Analysis")

st.write("Load saved data from the first page")

if 'df_to_save' not in st.session_state:
    st.session_state.df_to_save = None
if 'current_session_name' not in st.session_state:
    st.session_state.current_session_name = ""

st.subheader("1. Load Session Data")

data_dir = "temp"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
try:
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
except FileNotFoundError:
    available_files = []

if not available_files:
    st.warning("No `.dat` files found. Please go to the data loading page and save a session first.")
    st.stop()

selected_file = st.selectbox(
    "Select the data session to analyze:",
    options=available_files,
    help="Choose a `.dat` file saved from the data loading page."
)
def moving_average_filter(signal, M):
    y1 = np.convolve(signal, np.ones(M)/M, mode='same')
    y2 = np.convolve(y1, np.ones(M)/M, mode='same')
    return y2

def buat_tabel_rr_rrinterval(waktu_indices, peak_indices, fs):
    waktu_sec = waktu_indices / fs
    if len(peak_indices) < 2:
        return pd.DataFrame()
    interval_detik = np.diff(waktu_sec[peak_indices])
    bpm_per_beat = 60 / interval_detik
    beat_labels = [f"{waktu_sec[peak_indices[i]]:.2f}s - {waktu_sec[peak_indices[i+1]]:.2f}s" for i in range(len(peak_indices)-1)]
    df = pd.DataFrame({
        "Beat Interval (waktu)": beat_labels,
        "RR Interval (s)": interval_detik,
        "Respiratory Rate (BPM)": bpm_per_beat
    })
    df["RR Interval (s)"] = df["RR Interval (s)"].round(3)
    df["Respiratory Rate (BPM)"] = df["Respiratory Rate (BPM)"].round(2)
    return df

if st.button("Load and Analyze Data", key="LOADANALYZEDATA"):
    if selected_file:
        session_name = os.path.splitext(selected_file)[0]
        var = handler.load(f"temp/{session_name}")
        if var is not None:
            fs = 125
            analyzed = var.value
            analyzedtime = var.time
            st.session_state.analyzed = analyzed
            st.session_state.analyzedtime = analyzedtime
            st.session_state.fs = fs
            st.session_state.current_session_name = session_name
            st.session_state.df_to_save = None 
            st.success(f"Successfully loaded data from `{selected_file}`. You can now adjust filters and detect peaks below.")


analyzed = st.session_state.analyzed
analyzedtime = st.session_state.analyzedtime
fs = st.session_state.fs
session_name = st.session_state.current_session_name

st.divider()
st.subheader("2. Signal Filtering")
M = st.slider("Filter Window Size (M)", 1, 50, 8)
filtered_analyzed = moving_average_filter(analyzed, M)

fig_filter = go.Figure()
fig_filter.add_trace(go.Scatter(x=analyzedtime, y=analyzed, mode='lines', name='Original Signal', line=dict(color='rgba(23, 190, 207, 0.6)')))
fig_filter.add_trace(go.Scatter(x=analyzedtime, y=filtered_analyzed, mode='lines', name=f'Filtered Signal (M={M})', line=dict(color='rgba(214, 39, 40, 1.0)', width=2)))
fig_filter.update_layout(title_text='Original vs. Filtered Data', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_filter, use_container_width=True)

st.divider()
st.subheader("3. Peak Detection")
col1, col2 = st.columns(2)
prominence_val = col1.slider("Peak Prominence", 0.0, float(max(filtered_analyzed)), 0.5, 0.1)
distance_val = col2.slider("Minimum Distance Between Peaks (samples)", 1, 250, int(fs * 0.4))

idx_peaks, _ = find_peaks(filtered_analyzed, prominence=prominence_val, distance=distance_val)
val_peaks = filtered_analyzed[idx_peaks]

fig_peaks = go.Figure()
fig_peaks.add_trace(go.Scatter(x=analyzedtime, y=filtered_analyzed, mode='lines', name='Filtered Signal', line=dict(color='blue')))
fig_peaks.add_trace(go.Scatter(x=analyzedtime[idx_peaks], y=val_peaks, mode='markers', name='Detected Peaks', marker=dict(color='red', symbol='x', size=8)))
fig_peaks.update_layout(title='Detected Peaks on Filtered Signal', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_peaks, use_container_width=True)
st.info(f"**Total peaks detected:** {len(idx_peaks)}")
df_rr = buat_tabel_rr_rrinterval(analyzedtime, idx_peaks, fs)
st.session_state.df_to_save = df_rr

st.divider()
st.subheader("4. Calculated Intervals and Rates")
st.dataframe(st.session_state.df_to_save)

st.write("### Save Tachogram Data")
save_filename = st.text_input(
    "Enter filename for the tachogram data:",
    value=f"tachogram_{st.session_state.current_session_name}"
    )
        
if st.button("Save Data", key="SAVEDATAKEY"):
    handler.save(df_rr['RR Interval (s)'], df_rr['Respiratory Rate (BPM)'], f"temp/{save_filename}")