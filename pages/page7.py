import streamlit as st
from dwt_coeff import DWTCoeff
from deps import handler, updater
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import plotly.express as px
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

st.header("Vasomotor Activity")
st.write("Vasomotor Frequency Analysis  & Peak Vasomotor Magnitude")

try:
    available_files = [f for f in os.listdir('./temp/') if f.endswith('.dat')]
except FileNotFoundError:
    available_files = []

selected_file = st.selectbox(
    "Select the data session to analyze:",
    options=available_files
)
selected_file = selected_file.replace('.dat', '')

var = handler.load(f"temp/{selected_file}")
# const
fs = 50
dwt8 = var.value
timedwt8 = var.time / 50
total = len(dwt8)



st.markdown("---")
fig_filter = go.Figure()
fig_filter.add_trace(go.Scatter(x=timedwt8, y=dwt8, mode='lines', name='Vasomotor Signal', line=dict(color='rgba(23, 190, 207, 0.6)')))
#fig_filter.add_trace(go.Scatter(x=timedwt6, y=filtered_dwt6, mode='lines', name=f'Filtered Signal (M={M})', line=dict(color='rgba(214, 39, 40, 1.0)', width=2)))
fig_filter.update_layout(title_text='Vasomotor Signal', xaxis_title='Time / Index', yaxis_title='Amplitude', hovermode='x unified')
st.plotly_chart(fig_filter, use_container_width=True)


Y = fft(dwt8)
X = fftfreq(total, 1/fs)

Y_mag = 2.0/total * np.abs(Y[0:total//2])

# Get the corresponding positive frequencies
xf_pos = X[0:total//2]

MIN_MAGNITUDE = 1e-10 
Y_mag_clipped = np.clip(Y_mag, MIN_MAGNITUDE, None)
Y_db = 20 * np.log10(Y_mag_clipped)

peak_indices, _ = find_peaks(Y_db, prominence=10, height=-40) 
peak_freqs = xf_pos[peak_indices]
peak_amps_db = Y_db[peak_indices]
peak_data = {'Frequency (Hz)': peak_freqs.round(2), 'Amplitude (dB)': peak_amps_db.round(2)}

data = {
    'Frequency (Hz)': xf_pos,
    'Magnitude (dB)': Y_db
}

max_index = np.argmax(Y_db)
max_freq = xf_pos[max_index]
max_amp_db = Y_db[max_index]

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Peak Vasomotor Signal Frequency", value=f"{max_freq:.2f} Hz")
    
with col2:
    st.metric(label="Maximum Magnitude (dB)", value=f"{max_amp_db:.2f} dB")
    
fig_fft = px.line(data, x='Frequency (Hz)', y='Magnitude (dB)', 
                  title=f'Frequency Spectrum',
                  log_y=False,  
                  height=500)

st.markdown("---")
fig_fft.update_xaxes(range=[0, fs/2.0])
fig_fft.update_traces(mode='lines')

st.plotly_chart(fig_fft, use_container_width=True)

updater.save(max_freq, "maxVasoFreq")
updater.save(max_amp_db, "maxVasoMag")