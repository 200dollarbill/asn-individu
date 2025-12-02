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

# filtering method
# window size index must be odd
def savitzky_golay(signal, window_size, poly_order):
       
    n_points = len(signal)
    half_window = window_size // 2
    smoothed_signal = np.zeros(n_points)
    x_local = np.arange(-half_window, half_window + 1)
    X = np.vander(x_local, N=poly_order + 1, increasing=True)
    X_transpose = X.T
    XTX = np.dot(X_transpose, X)
    XTX_inv = np.linalg.inv(XTX)
    projection_matrix = np.dot(XTX_inv, X_transpose)
    weights = projection_matrix[0, :]
    for i in range(half_window, n_points - half_window):
        y_window = signal[i - half_window : i + half_window + 1]
        
        smoothed_value = np.dot(weights, y_window)
        
        smoothed_signal[i] = smoothed_value
    smoothed_signal[:half_window] = signal[:half_window]
    smoothed_signal[-half_window:] = signal[-half_window:]
    
    return smoothed_signal


fs = 2000
RT_FOOT = handler.load("footswitch")
RT_FOOT = RT_FOOT.value
LAT_G = handler.load("g_lateralis")
LAT_G = LAT_G.value
LAT_V = handler.load("v_lateralis")
INDEX = LAT_V.time
LAT_V = LAT_V.value


winsize = st.number_input(label="Window Size for filter (Must me odd/ganjil)", min_value=1, step=2,value=5)
poly_order = st.number_input(label="Polynomial order for filter (Must be under window size)", max_value=winsize, min_value=1, value=2)

filt_LAT_G = savitzky_golay(LAT_G,window_size=winsize,poly_order=poly_order)
filt_LAT_V = savitzky_golay(LAT_V,window_size=winsize,poly_order=poly_order)


st.write(INDEX)
st.set_page_config(layout="wide")
st.session_state.cwt_results = {}

st.subheader("Loaded Input Signals")
foot_plot = go.Figure()
foot_plot.add_trace(go.Scatter(x=INDEX, y=RT_FOOT, name='PCG Signal', mode='lines', line=dict(color='rgba(177, 255, 0, 0.8)')))

latg_plot = go.Figure()
latv_plot = go.Figure()
latg_plot.add_trace(go.Scatter(x=INDEX, y=LAT_G, name='Raw', mode='lines', line=dict(color='rgba(255, 185, 33, 0.3)')))
latg_plot.add_trace(go.Scatter(x=INDEX, y=filt_LAT_G, name='Filtered', mode='lines', line=dict(color='rgba(185, 255, 33, 0.8)')))
latv_plot.add_trace(go.Scatter(x=INDEX, y=LAT_V, name='Raw', mode='lines', line=dict(color='rgba(33, 185, 255, 0.3)')))
latv_plot.add_trace(go.Scatter(x=INDEX, y=filt_LAT_V, name='Filtered', mode='lines', line=dict(color='rgba(33, 255, 185, 0.8)')))
latg_plot.update_layout(title='Lateralis G', height=400)
latv_plot.update_layout(title='Lateralis V', height=400)
foot_plot.update_layout(title='Foot Plot', height=400)


st.plotly_chart(foot_plot, use_container_width=True)
st.plotly_chart(latg_plot, use_container_width=True)
st.plotly_chart(latv_plot, use_container_width=True)

if st.button("Apply CWT"):
    cwt_magnitude_lv, freqs = cwt(filt_LAT_V, fs)
    cwt_magnitude_lg, freqs = cwt(filt_LAT_G, fs)

    st.session_state.cwt_results = {'freqs': freqs, 
                                    'lv_mag' : cwt_magnitude_lv, 
                                    'lg_mag' : cwt_magnitude_lg
                                    }

    latv_cwt = st.session_state.cwt_results['lv_mag']
    latg_cwt = st.session_state.cwt_results['lg_mag']

    cwt_freqs = st.session_state.cwt_results['freqs']


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

    with st.spinner("Calculating CWT..."):
        cwt_magnitude_lv, freqs = cwt(LAT_V, fs)
        cwt_magnitude_lg, freqs = cwt(LAT_G, fs)

        st.session_state.cwt_results = {
            'freqs': freqs, 
            'lv_mag' : cwt_magnitude_lv, 
            'lg_mag' : cwt_magnitude_lg
        }

    latv_cwt = st.session_state.cwt_results['lv_mag']
    latg_cwt = st.session_state.cwt_results['lg_mag']
    cwt_freqs = st.session_state.cwt_results['freqs']

    def plot_3d_cwt(z_data, x_data, y_data, title):
        step_x = 5 
        step_y = 1  
        
        fig = go.Figure(data=[go.Surface(
            z=z_data[::step_y, ::step_x], 
            x=x_data[::step_x], 
            y=y_data[::step_y],
            colorscale='Jet',
            contours_z=dict(
                show=True, usecolormap=True, 
                highlightcolor="limegreen", project_z=True
            )
        )])

        fig.update_layout(
            title=title,
            autosize=True,
            height=700,
            scene=dict(
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                zaxis_title='Magnitude',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2) 
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        return fig

    st.subheader("3D CWT Scalograms")

    st.write("ateralis G (3D Surface)")
    fig_lg_3d = plot_3d_cwt(latg_cwt, INDEX, cwt_freqs, "Lateralis G - 3D CWT")
    st.plotly_chart(fig_lg_3d, use_container_width=True)

    st.write("Lateralis V (3D Surface)")
    fig_lv_3d = plot_3d_cwt(latv_cwt, INDEX, cwt_freqs, "Lateralis V - 3D CWT")
    st.plotly_chart(fig_lv_3d, use_container_width=True)
