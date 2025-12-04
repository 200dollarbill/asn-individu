import streamlit as st
import numpy as np
from deps import handler
import plotly.graph_objects as go
import pywt
from scipy.signal import find_peaks

def cwt(signal_data, fs, f_min, f_max):
    wavelet_name = 'cmor2.5-0.5' 

    center_freq = pywt.central_frequency(wavelet_name)
    min_scale = (center_freq * fs) / f_max
    max_scale = (center_freq * fs) / f_min
    # scales = np.linspace(min_scale, max_scale, 128)
    scales = np.geomspace(min_scale, max_scale, num=128)
    # scales = np.arange(1, 256)
    coeffs, freqs = pywt.cwt(signal_data, scales, wavelet_name, sampling_period=1.0/fs)
    cwt_magnitude = np.abs(coeffs)
    return cwt_magnitude, freqs

g_count = 0
l_count = 0


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


def thresholded_cwt_binary(Z, thr_percent=5.0, method='range'):
    Z = np.asarray(Z, dtype=float)
    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)

    if method == 'max':
        thr = (thr_percent / 100.0) * zmax
    else:
        thr = zmin + (thr_percent/100.0) * (zmax - zmin)

    mask = Z > thr
    return mask, thr

def get_segments_from_mask(mask, time_index):
    active = np.any(mask, axis=0).astype(int)

    segments = []
    start = None

    for i in range(len(active)):
        if active[i] == 1 and start is None:
            start = i
        if active[i] == 0 and start is not None:
            segments.append((start, i-1))
            start = None

    if start is not None:
        segments.append((start, len(active)-1))

    segments_time = [(time_index[s], time_index[e]) for s, e in segments]
    return segments_time

def label_GL_segment(seg, index):
    start, end = seg 
    
    mid_time = (start + end) / 2.0
    
    min_time = index[0]
    max_time = index[-1]
    duration = max_time - min_time
    
    if duration == 0: return "Unknown"

    mid_pct = ((mid_time - min_time) / duration) * 100
    if 4 <= mid_pct <= 14:  return "GL1"
    if 21 <= mid_pct <= 45: return "GL2"
    if 83 <= mid_pct <= 100: return "GL3"
    
    print(f'gl mid {mid_pct} start {start} end {end}')
    return "Unknown"

def label_VL_segment(seg,index):
    start, end = seg
    mid_time = (start + end) / 2.0
    
    min_time = index[0]
    max_time = index[-1]
    duration = max_time - min_time
    if duration == 0: return "Unknown"
    mid_pct = ((mid_time - min_time) / duration) * 100
    if 0 <= mid_pct <= 15:   return "VL1"
    if 33 <= mid_pct <= 42:  return "VL2"
    if 72 <= mid_pct <= 78:  return "VL3"
    if 84 <= mid_pct <= 100: return "VL4"
    print(f'vl mid {mid_pct} start {start} end {end}')
    return "Unknown"


def plot_segmentation_plotly(cwt_data, time_axis, freqs, segments, labels, title, threshold_val):
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=cwt_data, 
        x=time_axis, 
        y=freqs,
        colorscale='Jet',
        colorbar=dict(title='Magnitude')
    ))

    for (start_t, end_t), label in zip(segments, labels):
        fig.add_vline(x=start_t, line_width=2, line_dash="dash", line_color="red")
        fig.add_vline(x=end_t, line_width=2, line_dash="dash", line_color="cyan")
        
        mid_t = (start_t + end_t) / 2
        fig.add_annotation(
            x=mid_t, 
            y=freqs[-1], 
            text=label,
            showarrow=False,
            font=dict(color="white", size=12, shadow="2px 2px 2px black"),
            bgcolor="rgba(0,0,0,0.5)"
        )

    fig.update_layout(
        title=f"{title} (Threshold > {threshold_val:.2f})",
        xaxis_title='Time / Gait', 
        yaxis_title='Frequency (Hz)',
        height=500
    )
    return fig

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

st.subheader("CWT & Segmentation Settings")
col_t1, col_t2 = st.columns(2)
with col_t1:
    thr_gl_pct = st.slider("GL Threshold %", 0, 100, 27)
with col_t2:
    thr_vl_pct = st.slider("VL Threshold %", 0, 100, 22)

if st.button("Apply CWT"):
    cwt_magnitude_lv, freqs = cwt(filt_LAT_V, fs,f_min=0.1, f_max=250)
    cwt_magnitude_lg, freqs = cwt(filt_LAT_G, fs,f_min=0.1, f_max=250)

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
        title='Lateralis G Scalogram',xaxis_title='Time', yaxis_title='Frequency',yaxis=dict(type='linear'))


    latv_cwt_plot = go.Figure()
    latv_cwt_plot.add_trace(go.Heatmap(
        z=latv_cwt, x=INDEX, y=cwt_freqs,colorscale='Jet', colorbar=dict(title='Magnitude')
    ))
    latv_cwt_plot.update_layout(
        title='Lateralis V Scalogram',xaxis_title='Time', yaxis_title='Frequency',yaxis=dict(type='linear'))



    st.plotly_chart(latg_cwt_plot, use_container_width=True)
    st.plotly_chart(latv_cwt_plot, use_container_width=True)
    freq_band = (cwt_freqs >= 20) & (cwt_freqs <= 150)

    with st.spinner("Calculating CWT..."):
        cwt_magnitude_lv, freqs = cwt(LAT_V, fs,f_min=0.1, f_max=250)
        cwt_magnitude_lg, freqs = cwt(LAT_G, fs,f_min=0.1,f_max=250)

        st.session_state.cwt_results = {
            'freqs': freqs, 
            'lv_mag' : cwt_magnitude_lv, 
            'lg_mag' : cwt_magnitude_lg
        }

    latv_cwt = st.session_state.cwt_results['lv_mag']
    latg_cwt = st.session_state.cwt_results['lg_mag']
    cwt_freqs = st.session_state.cwt_results['freqs']

    mask_GL, val_thr_GL = thresholded_cwt_binary(latg_cwt, thr_percent=thr_gl_pct, method='range')
    segs_GL_time = get_segments_from_mask(mask_GL, INDEX)
    labels_GL = [label_GL_segment(seg, INDEX) for seg in segs_GL_time]
    mask_VL, val_thr_VL = thresholded_cwt_binary(latv_cwt, thr_percent=thr_vl_pct, method='range')
    segs_VL_time = get_segments_from_mask(mask_VL, INDEX)
    labels_VL = [label_VL_segment(seg, INDEX) for seg in segs_VL_time]

    st.subheader("2D CWT with Segmentation")
    
    col_cwt1, col_cwt2 = st.columns(2)
    
    with col_cwt1:
        fig_seg_gl = plot_segmentation_plotly(
            latg_cwt, INDEX, cwt_freqs, segs_GL_time, labels_GL, "Lateralis G Activations", val_thr_GL
        )
        st.plotly_chart(fig_seg_gl, use_container_width=True)
        
        st.write("**GL Activations:**")
        for (s, e), lab in zip(segs_GL_time, labels_GL):
            st.caption(f"- {lab}: {s:.2f} to {e:.2f}")

    with col_cwt2:
        fig_seg_vl = plot_segmentation_plotly(
            latv_cwt, INDEX, cwt_freqs, segs_VL_time, labels_VL, "Lateralis V Activations", val_thr_VL
        )
        st.plotly_chart(fig_seg_vl, use_container_width=True)
        
        st.write("**VL Activations:**")
        for (s, e), lab in zip(segs_VL_time, labels_VL):
            st.caption(f"- {lab}: {s:.2f} to {e:.2f}")

    st.subheader("3D CWT Scalograms")

    st.write("ateralis G (3D Surface)")
    fig_lg_3d = plot_3d_cwt(latg_cwt, INDEX, cwt_freqs, "Lateralis G - 3D CWT")
    st.plotly_chart(fig_lg_3d, use_container_width=True)

    st.write("Lateralis V (3D Surface)")
    fig_lv_3d = plot_3d_cwt(latv_cwt, INDEX, cwt_freqs, "Lateralis V - 3D CWT")
    st.plotly_chart(fig_lv_3d, use_container_width=True)
