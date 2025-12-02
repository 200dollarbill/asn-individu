import streamlit as st
import numpy as np
from deps import handler
import plotly.graph_objects as go
import pywt
from scipy.signal import find_peaks

def cwt(signal_data, fs):
    scales = np.arange(1, 256)
    wavelet_name = 'cmor1.5-1.0' 
    coeffs, freqs = pywt.cwt(signal_data, scales, wavelet_name, sampling_period=1.0/fs)
    cwt_magnitude = np.abs(coeffs)
    return cwt_magnitude, freqs

fs = 2000
RT_FOOT = handler.load("footswitch")
RT_FOOT = RT_FOOT.value
LAT_G = handler.load("g_lateralis")
LAT_G = LAT_G.value
LAT_V = handler.load("v_lateralis")
INDEX = LAT_V.time
LAT_V = LAT_V.value


st.write(INDEX  )
st.set_page_config(layout="wide")
st.session_state.cwt_results = {}

st.subheader("Loaded Input Signals")
foot_plot = go.Figure()
foot_plot.add_trace(go.Scatter(x=INDEX, y=RT_FOOT, name='PCG Signal', mode='lines', line=dict(color='rgba(177, 255, 0, 0.8)')))

latg_plot = go.Figure()
latv_plot = go.Figure()
latg_plot.add_trace(go.Scatter(x=INDEX, y=LAT_G, name='ECG Signal', mode='lines', line=dict(color='rgba(255, 185, 33, 0.8)')))
latv_plot.add_trace(go.Scatter(x=INDEX, y=LAT_V, name='ECG Signal', mode='lines', line=dict(color='rgba(33, 185, 255, 0.8)')))
latg_plot.update_layout(title='Lateralis G', height=400)
latv_plot.update_layout(title='Lateralis V', height=400)
foot_plot.update_layout(title='Foot Plot', height=400)

st.plotly_chart(latg_plot, use_container_width=True)
st.plotly_chart(latv_plot, use_container_width=True)
st.plotly_chart(foot_plot, use_container_width=True)

if st.button("Apply CWT"):
    cwt_magnitude_lv, freqs = cwt(LAT_V, fs)
    # cwt_magnitude_foots, freqs = cwt(RT_FOOT, fs)
    cwt_magnitude_lg, freqs = cwt(LAT_G, fs)

    st.session_state.cwt_results = {'freqs': freqs, 
                                    'lv_mag' : cwt_magnitude_lv, 
                                    'lg_mag' : cwt_magnitude_lg
                                    }

    latv_cwt = st.session_state.cwt_results['lv_mag']
    latg_cwt = st.session_state.cwt_results['lg_mag']

    cwt_freqs = st.session_state.cwt_results['freqs']

    # threshold_percent = st.slider(
    #     "Magnitude Threshold (%)", min_value=0, max_value=100, value=85,
    #     )

    # threshold_value = np.percentile(cwt_mag, threshold_percent)

    # foot_cwt_plot = go.Figure()
    # foot_cwt_plot.add_trace(go.Heatmap(
    #     z=foot_cwt, x=INDEX, y=cwt_freqs,colorscale='Jet', colorbar=dict(title='Magnitude')
    # ))
    # foot_cwt_plot.update_layout(
    #     title='CWT Scalogram',xaxis_title='Time', yaxis_title='Frequency',yaxis=dict(type='linear'))

    latg_cwt_plot = go.Figure()
    latg_cwt_plot.add_trace(go.Heatmap(
        z=latg_cwt, x=INDEX, y=cwt_freqs,colorscale='Jet', colorbar=dict(title='Magnitude')
    ))
    latg_cwt_plot.update_layout(
        title='CWT Scalogram',xaxis_title='Time', yaxis_title='Frequency',yaxis=dict(type='linear'))


    latv_cwt_plot = go.Figure()
    latv_cwt_plot.add_trace(go.Heatmap(
        z=latv_cwt, x=INDEX, y=cwt_freqs,colorscale='Jet', colorbar=dict(title='Magnitude')
    ))
    latv_cwt_plot.update_layout(
        title='CWT Scalogram',xaxis_title='Time', yaxis_title='Frequency',yaxis=dict(type='linear'))



    st.plotly_chart(latg_cwt_plot, use_container_width=True)
    st.plotly_chart(latv_cwt_plot, use_container_width=True)
    freq_band = (cwt_freqs >= 20) & (cwt_freqs <= 150)

