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

if 'MAX_FREQ' not in st.session_state:
    st.session_state.MAX_FREQ = 500

if 'time_axis' not in st.session_state:
    st.session_state.time_axis = 0

if 'sliced_spectrogram' not in st.session_state:
    st.session_state.sliced_spectogram = 0

if 'sliced_freq_axis' not in  st.session_state:
    st.session_state.sliced_freq_axis = 0

if st.button("Calculate STFT"):
    st.session_state.LAT_V = handler.load(f"filtered_LAT_V")
    st.session_state.LAT_G = handler.load(f"filtered_LAT_G")
    st.session_state.RT_FOOT = handler.load(f"RT_FOOT")


def stft(input):
    MAX_FREQ_HZ = st.session_state.MAX_FREQ
    stft_config = config.create_stft_instance()
    x_stft, start_list, stop_list = stft_config.stft(input)

    magnitude_spectrogram = np.abs(x_stft)
    db_spectrogram = 20 * np.log10(magnitude_spectrogram + 1e-9)

    num_frames = x_stft.shape[1]
    segment_len = config.segment_length 
    st.session_state.time_axis = (start_list[:num_frames] + segment_len / 2) / Fs

    num_freq_bins = x_stft.shape[0]
    freq_axis = np.linspace(0, Fs / 2, num_freq_bins)

    cutoff_index = np.where(freq_axis >= MAX_FREQ_HZ)[0][0]

    st.session_state.sliced_spectrogram = magnitude_spectrogram[:cutoff_index, :]
    st.session_state.sliced_freq_axis = freq_axis[:cutoff_index]        

INDEX = st.session_state.LAT_V.time
LAT_V = st.session_state.LAT_V.value
LAT_G = st.session_state.LAT_G.value
RT_FOOT = st.session_state.RT_FOOT.value
Fs = 2000
st.session_state.MAX_FREQ = st.number_input(label="Max Frequency",value=500)  



window_count = st.number_input("window", value=300)
overlap = st.number_input("overlap percentage", value=90)  
config = STFTConfigurator(
    signal_duration_seconds=len(INDEX)/Fs,
    sampling_rate=2000,
    overlap_percentage=overlap,
    window_count=window_count,
    window_function=hann
)

stft_latg = stft(LAT_G)

sliced_spectrogram = st.session_state.sliced_spectrogram
sliced_freq_axis = st.session_state.sliced_freq_axis
time_axis = st.session_state.time_axis

fig0 = go.Figure(data=go.Heatmap(
    z=sliced_spectrogram, 
    x=time_axis,
    y=sliced_freq_axis,
    colorscale='Jet',
    colorbar=dict(title='Magnitude (dB)') 
))

fig0.update_layout(
    title=f'Lateralis G Spectrogram',
    xaxis_title='Time (s)',
    yaxis_title='Frequency (Hz)'
)

st.plotly_chart(fig0, use_container_width=True)


stft_latv = stft(LAT_V)


sliced_spectrogram = st.session_state.sliced_spectrogram
sliced_freq_axis = st.session_state.sliced_freq_axis
time_axis = st.session_state.time_axis

fig = go.Figure(data=go.Heatmap(
    z=sliced_spectrogram, 
    x=time_axis,
    y=sliced_freq_axis,
    colorscale='Jet',
    colorbar=dict(title='Magnitude (dB)') 
))

fig.update_layout(
    title=f'Lateralis V Spectrogram',
    xaxis_title='Time (s)',
    yaxis_title='Frequency (Hz)'
)

st.plotly_chart(fig, use_container_width=True)


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=INDEX, y=LAT_G, mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)', width=0.8)))
fig2.update_layout(title='Lateralis G',height=500, width=1200)    

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=INDEX, y=LAT_V, mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)', width=0.8)))
fig3.update_layout(title='Lateralis V',height=500, width=1200) 


st.plotly_chart(fig2, use_container_width=True)     
st.plotly_chart(fig3, use_container_width=True)

signal_duration = len(LAT_G)/50
st.write(signal_duration)
frames_per_second = len(time_axis) / signal_duration

