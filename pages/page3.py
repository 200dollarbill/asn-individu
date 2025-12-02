import numpy as np
import streamlit as st
from deps import handler
import plotly.graph_objects as go
from ft import STFT_LIB, STFTConfigurator
from scipy.signal.windows import hann
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.ndimage import label

if 'LAT_V' not in st.session_state:
    st.session_state.LAT_V = 'test'

if 'LAT_G' not in st.session_state:
    st.session_state.LAT_G = 'test'

if 'RT_FOOT' not in st.session_state:
    st.session_state.RT_FOOT = 'test'


if st.button("Calculate STFT"):
    st.session_state.LAT_V = handler.load(f"v_lateralis")
    st.session_state.LAT_G = handler.load(f"g_lateralis")
    st.session_state.RT_FOOT = handler.load(f"footswitch")
        

INDEX = st.session_state.LAT_V.time
LAT_V = st.session_state.LAT_V.value
LAT_G = st.session_state.LAT_G.value
RT_FOOT = st.session_state.RT_FOOT.value
Fs = 2000

config = STFTConfigurator(
    signal_duration_seconds=len(INDEX)/Fs,
    sampling_rate=2000,
    overlap_percentage=90,
    window_count=300,
    window_function=hann
)

stft_config = config.create_stft_instance()


x_stft, start_list, stop_list = stft_config.stft(LAT_G)

magnitude_spectrogram = np.abs(x_stft)
db_spectrogram = 20 * np.log10(magnitude_spectrogram + 1e-9)

num_frames = x_stft.shape[1]
segment_len = config.segment_length 
time_axis = (start_list[:num_frames] + segment_len / 2) / Fs

num_freq_bins = x_stft.shape[0]
freq_axis = np.linspace(0, Fs / 2, num_freq_bins)

MAX_FREQ_HZ = st.number_input(label="Max Frequency")  

cutoff_index = np.where(freq_axis >= MAX_FREQ_HZ)[0][0]

sliced_spectrogram = magnitude_spectrogram[:cutoff_index, :]
sliced_freq_axis = freq_axis[:cutoff_index]

fig = go.Figure(data=go.Heatmap(
    z=sliced_spectrogram, 
    x=time_axis,
    y=sliced_freq_axis,
    colorscale='Jet',
    colorbar=dict(title='Magnitude (dB)') 
))

fig.update_layout(
    title=f'PCG Spectrogram (STFT) up to {MAX_FREQ_HZ} Hz',
    xaxis_title='Time (s)',
    yaxis_title='Frequency (Hz)'
)

st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=INDEX, y=LAT_G, mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)', width=0.8)))
fig2.update_layout(title='PCG',height=500, width=1200)        
st.plotly_chart(fig2, use_container_width=True)

signal_duration = len(LAT_G)/50
st.write(signal_duration)
frames_per_second = len(time_axis) / signal_duration

