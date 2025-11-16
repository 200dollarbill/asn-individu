import numpy as np
import streamlit as st
from deps import handler
import plotly.graph_objects as go
from ft import STFT_LIB, STFTConfigurator
from scipy.signal.windows import hann





with st.form(key='input'):
    name = st.text_input("Input Thresholded Data Name", )
    
    input_button = st.form_submit_button(label="Load Data")
    clear_button = st.form_submit_button(label="Clear All")

if input_button:
    st.session_state.show_input = True
    st.session_state.cwt_results = None 
    st.session_state.pcg_data = handler.load(f"data/{name}PCG")
    st.session_state.ecg_data = handler.load(f"data/{name}ECG")

pcg = st.session_state.pcg_data
pcg_time = pcg.time
pcg_value = pcg.value

ecg = st.session_state.ecg_data
ecg_time = ecg.time
ecg_value = ecg.value
Fs = 2000

config = STFTConfigurator(
    signal_duration_seconds=len(pcg_time)/Fs,
    sampling_rate=2000,
    overlap_percentage=90,
    window_count=200,
    window_function=hann
)

stft_config = config.create_stft_instance()


x_stft, start_list, stop_list = stft_config.stft(pcg_value)

#st.write(x_stft, start_list,stop_list)

magnitude_spectrogram = np.abs(x_stft)
db_spectrogram = 20 * np.log10(magnitude_spectrogram + 1e-9)

num_frames = x_stft.shape[1]
segment_len = config.segment_length 
time_axis = (start_list[:num_frames] + segment_len / 2) / Fs

num_freq_bins = x_stft.shape[0]
freq_axis = np.linspace(0, Fs / 2, num_freq_bins)

MAX_FREQ_HZ = 125  

cutoff_index = np.where(freq_axis >= MAX_FREQ_HZ)[0][0]

# change to magnitude / db
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


# fig2 = go.Figure(data=go.chart(
#     x=time_axis,
#     y=sliced_freq_axis,
#     colorscale='Jet',
#     colorbar=dict(title='Magnitude (dB)') # Add a label for the color bar
# ))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=pcg_time, y=pcg_value, mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)', width=0.8)))
fig2.update_layout(title='PCG',height=500, width=1200)        
st.plotly_chart(fig2, use_container_width=True)

