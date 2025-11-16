import numpy as np
import streamlit as st
from deps import handler
import plotly.graph_objects as go
from ft import STFT_LIB, STFTConfigurator
from scipy.signal.windows import hann
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.ndimage import label

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

st.subheader("COG Detection from STFT")

col1, col2 = st.columns(2)
with col1:
    threshold_percent = st.slider("Magnitude Threshold (%)", 0, 100, 90, key="stft_thresh")
with col2:
    min_area_px = st.slider("Minimum Area (pixels)", 1, 500, 20, key="stft_area")

threshold_value = np.percentile(db_spectrogram, threshold_percent)
binary_mask = db_spectrogram > threshold_value
labeled_regions, num_labels = label(binary_mask)
filtered_mask = np.zeros_like(binary_mask)
for i in range(1, num_labels + 1):
    if np.sum(labeled_regions == i) >= min_area_px:
        filtered_mask[labeled_regions == i] = True

masked_spectrogram = db_spectrogram.copy()
masked_spectrogram[~filtered_mask] = np.min(db_spectrogram)
energy_signal = np.sum(masked_spectrogram, axis=0)

signal_duration = len(pcg_time)/50
st.write(signal_duration)
frames_per_second = len(time_axis) / signal_duration
peak_distance_frames = max(1, int(0.05 * frames_per_second))
peaks, _ = find_peaks(energy_signal, height=np.percentile(energy_signal, 85), distance=peak_distance_frames)

refined_peak_indices = []
if len(peaks) > 0:
    for p_index in peaks:
        start_sample = start_list[p_index]
        stop_sample = stop_list[p_index]
        
        search_window = pcg_value[start_sample:stop_sample]
        
        local_peak_index = np.argmax(np.abs(search_window))
        
        refined_peak_indices.append(start_sample + local_peak_index)


st.write(f"Detected {len(refined_peak_indices)} COGs based on the criteria.")
fig_peaks = go.Figure()
fig_peaks.add_trace(go.Scatter(
    x=pcg.time, y=pcg_value, name='PCG Signal', mode='lines',
    line=dict(color='rgba(255, 176, 0, 0.7)')
))

if refined_peak_indices:
    fig_peaks.add_trace(go.Scatter(
        x=pcg.time[refined_peak_indices], y=pcg_value[refined_peak_indices], 
        name='Detected COGs', mode='markers',
        marker=dict(color='red', size=10, symbol='x')
    ))

fig_peaks.update_layout(title='PCG Signal with Refined COGs from STFT', height=400)
st.plotly_chart(fig_peaks, use_container_width=True)
sliced_mask = filtered_mask[:cutoff_index, :]

highlight_z = np.full(sliced_mask.shape, np.nan)
highlight_z[sliced_mask] = 1

highlight_colorscale = [
    [0, 'rgba(0,0,0,0.5)'],
    [1, 'rgba(10, 255, 255, 1)'] 
]

fig_stft = go.Figure()

fig_stft = go.Figure(data=go.Heatmap(
    z=sliced_spectrogram, 
    x=time_axis,
    y=sliced_freq_axis,
    colorscale='Jet',
    colorbar=dict(title='Magnitude (dB)') 
))

fig_stft.add_trace(go.Heatmap(
    z=highlight_z,
    x=time_axis,
    y=sliced_freq_axis,
    colorscale=highlight_colorscale,
    showscale=False, 
    name='Detected Region',
    hoverinfo='none' 
))

st.plotly_chart(fig_stft, use_container_width=True)
st.write(f"Detected {len(refined_peak_indices)} COGs based on the criteria.")
fig_peaks = go.Figure()
