import streamlit as st
import numpy as np
from deps import handler
import plotly.graph_objects as go
import pywt
from scipy.signal import find_peaks

def perform_cwt_analysis(signal_data, fs):
    scales = np.arange(1, 256)
    wavelet_name = 'cmor1.5-1.0' 
    coeffs, freqs = pywt.cwt(signal_data, scales, wavelet_name, sampling_period=1.0/fs)
    
    cwt_magnitude = np.abs(coeffs)
    
    return cwt_magnitude, freqs


st.set_page_config(layout="wide")

if 'show_input' not in st.session_state:
    st.session_state.show_input = False
if 'cwt_results' not in st.session_state:
    st.session_state.cwt_results = None
if 'pcg_data' not in st.session_state:
    st.session_state.pcg_data = None


with st.form(key='input'):
    name = st.text_input("Input Thresholded Data Name", )
    
    input_button = st.form_submit_button(label="Load Data")
    clear_button = st.form_submit_button(label="Clear All")

if input_button:
    st.session_state.show_input = True
    st.session_state.cwt_results = None 
    st.session_state.pcg_data = handler.load(f"data/{name}PCG")
    st.session_state.ecg_data = handler.load(f"data/{name}ECG")

if st.session_state.show_input:
    pcg = st.session_state.pcg_data
    ecg = st.session_state.ecg_data
    
    pcgtime = pcg.time
    pcgval = pcg.value
    ecgtime = ecg.time
    ecgval = ecg.value
    fs = 2000

    st.subheader("Loaded Input Signals")
    fig_input = go.Figure()
    fig_input.add_trace(go.Scatter(x=pcgtime, y=pcgval, name='PCG Signal', mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)')))
    fig_input.add_trace(go.Scatter(x=ecgtime, y=ecgval, name='ECG Signal', mode='lines', line=dict(color='rgba(33, 185, 33, 0.8)')))
    fig_input.update_layout(title='Loaded ECG & PCG Data', height=400)
    st.plotly_chart(fig_input, use_container_width=True)
    
    if st.button("Apply CWT"):
        cwt_magnitude, freqs = perform_cwt_analysis(pcgval, fs)
        st.session_state.cwt_results = {'magnitude': cwt_magnitude, 'freqs': freqs}

    if st.session_state.cwt_results is not None:
        cwt_mag = st.session_state.cwt_results['magnitude']
        cwt_freqs = st.session_state.cwt_results['freqs']

        threshold_percent = st.slider(
            "Magnitude Threshold (%)", min_value=0, max_value=100, value=85,
          )
        
        threshold_value = np.percentile(cwt_mag, threshold_percent)

        fig_cwt = go.Figure()
        
        fig_cwt.add_trace(go.Heatmap(
            z=cwt_mag, x=pcgtime, y=cwt_freqs,colorscale='Jet', colorbar=dict(title='Magnitude')
        ))
        
        fig_cwt.add_trace(go.Contour(
            z=cwt_mag, x=pcgtime, y=cwt_freqs,contours_coloring='lines',line_color='white',line_width=2,showscale=False,contours=dict(start=threshold_value,end=threshold_value,size=0 )
        ))
        
        fig_cwt.update_layout(
            title='CWT Scalogram',xaxis_title='Time', yaxis_title='Frequency',yaxis=dict(type='log'))
        st.plotly_chart(fig_cwt, use_container_width=True)
        freq_band = (cwt_freqs >= 20) & (cwt_freqs <= 150)
        energy_signal = np.sum(cwt_mag[freq_band, :], axis=0)
        
        peak_height_threshold = np.percentile(energy_signal, threshold_percent)
        peaks, _ = find_peaks(energy_signal, height=peak_height_threshold, distance=fs*0.1)

        st.write(f"{threshold_percent}% threshold.")

        fig_peaks = go.Figure()
        fig_peaks.add_trace(go.Scatter(x=pcgtime, y=pcgval, name='PCG Signal', mode='lines', line=dict(color='rgba(255, 176, 0, 0.7)')))
        fig_peaks.add_trace(go.Scatter(x=pcgtime[peaks], y=pcgval[peaks], name='Detected S1/S2 Peaks', mode='markers',
                                       marker=dict(color='red', size=10, symbol='x')))
        fig_peaks.update_layout(title='PCG Signal and peaks from cwt', height=400)
        st.plotly_chart(fig_peaks, use_container_width=True)

if clear_button:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()