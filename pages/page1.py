import streamlit as st
import numpy as np
from deps import handler
from pan_tompkins import Pan_Tompkins_QRS, HeartRate
import plotly.graph_objects as go
import wfdb


fig = go.Figure()

pcgY=np.arange(1)
cutecgval= np.arange(1)
cutecgtime=np.arange(1)
cutpcgval= np.arange(1)
cutpcgtime=np.arange(1)


st.set_page_config(layout="wide")
if 'show_threshold' not in st.session_state:
    st.session_state.show_threshold = False
if 'show_input' not in st.session_state:
    st.session_state.show_input = False
if 'thresholded' not in st.session_state:
    st.session_state.thresholded = False

with st.form(key='input'):
    name = st.text_input("Input Data","4")
    input = st.form_submit_button(label="Load Data")
    clear = st.form_submit_button(label="Clear")


def replot(arrx,arry, color, mode,limiter,normalize):
    fig.add_trace(go.Scatter(x=arrx[limiter:]/fs, y=arry[limiter:]/normalize, mode=mode, line=dict(color=color, width=0.8)))


if input or st.session_state.show_input:
    if int(name) > 9:
        record = wfdb.rdrecord('data/a00'+str(name))
        
    else:
        record = wfdb.rdrecord('data/a000'+str(name)) 
    name = str(name)

    pcgY = record.p_signal[:,0]
    ecgY = record.p_signal[:,1]
    pcgX = np.arange(record.p_signal.shape[0])
    ecgX = np.arange(record.p_signal.shape[0])
    # print(ecgX)
    # print(pcgX)

    signal_data = record.p_signal
    st.write(f"Signal data name : a000{name}.dat")
    st.write(f"Signal data shape: {signal_data.shape}")
    st.write(f"Sampling frequency: {record.fs} Hz")
    st.write(f"Signal names: {record.sig_name}")
    st.write(f"Signal units: {record.units}")
    st.write(f"Record comments: {record.comments}")
    handler.save(ecgX,ecgY,filename="data/ecg"+name)
    handler.save(pcgX,pcgY,filename="data/pcg"+name)

    #wfdb.plot_wfdb(record=record,time_units='seconds')
    fs = 2000 
    ecgraw = handler.load(f"data/ecg{name}")
    ecgtime = ecgraw.time
    ecgval = ecgraw.value

    pcgraw = handler.load(f"data/pcg{name}")
    pcgval = pcgraw.value
    pcgtime = pcgraw.time

    # type for each = ndarray
    # shape = (71193,)
    # st.write(ecgtime.shape)
    qrs_detector = Pan_Tompkins_QRS()
    bpass_signal, _, _, mwin_signal = qrs_detector.solve(ecgval, fs)

    hr_detector = HeartRate(raw_signal=ecgval, mwin_signal=mwin_signal, bpass_signal=bpass_signal, samp_freq=fs)
    r_peaks = hr_detector.find_r_peaks()
    r_peaks = r_peaks[r_peaks > (0.5 * fs)] 

    # st.write("r peaks at:", r_peaks)
    rr_intervals = np.diff(r_peaks)
    avg_rr_interval = np.mean(rr_intervals)

    heart_rate_bpm = (fs * 60) / avg_rr_interval
    st.write(f"Heart Rate: {heart_rate_bpm:.2f}  BPM")

    # replot = add trace but better (for lines)
    replot(ecgtime,ecgval,'rgba(255, 176, 0, 0.8)', 'lines', 3000,1)
    fig.add_trace(go.Scatter(x=r_peaks[2:]/fs, y=ecgval[r_peaks[2:]], mode='markers', 
                            marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='Pan Tompkins Result',height=500, width=1200)
    
    st.plotly_chart(fig)
    
    fig.data = []
    replot(pcgtime,pcgval,'rgba(125, 100, 0, 0.9)', 'lines', 3000,6000) 
    fig.update_layout(title='PCG Data',height=500, width=1200)
    st.plotly_chart(fig)
    
    fig.data = []
    replot(ecgtime,ecgval,'rgba(125, 100, 0, 0.9)', 'lines', 3000,1)
    replot(pcgtime,pcgval,'rgba(125, 100, 0, 0.9)', 'lines', 3000,6000)
    fig.add_trace(go.Scatter(x=r_peaks[2:]/fs, y=ecgval[r_peaks[2:]], mode='markers', 
                            marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='PCG Data',height=500, width=1200)
    st.plotly_chart(fig)
    
    st.session_state.show_threshold = True
    st.session_state.show_input = True
if clear:
    st.session_state.show_threshold = False
    st.session_state.show_input = False
    st.rerun()

if st.session_state.show_threshold:
    with st.form(key='threshold'):
        threshold = st.slider(label="Threshold (seconds)", min_value=float(0), max_value=len(pcgY)/fs,value=(0.0,2.0))        
        cut = st.form_submit_button(label="Threshold")
        save = st.form_submit_button(label="Plot")
        savedname = st.text_input(label="Saved cut File Name (+ECG/PCG)")
    if cut or st.session_state.thresholded:       
        cutecgtime = ecgtime[int(threshold[0]*fs):int(threshold[1]*fs)]
        cutecgtime = cutecgtime-int(threshold[0]*fs)
        cutecgval = ecgval[int(threshold[0]*fs):int(threshold[1]*fs)]
        cutpcgtime = pcgtime[int(threshold[0]*fs):int(threshold[1]*fs)]
        cutpcgtime = cutpcgtime-int(threshold[0]*fs)
        cutpcgval = pcgval[int(threshold[0]*fs):int(threshold[1]*fs)]    
        fig.data = []
     
        fig.add_trace(go.Scatter(x=cutecgtime/fs, y=cutecgval, mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)', width=0.8)))
        fig.update_layout(title='Thresholded result',height=500, width=1200)
        st.plotly_chart(fig)
        st.session_state.thresholded = True
    if save:
        st.write(cutecgtime)
        handler.save(cutecgtime,cutecgval,f"data/{savedname}ECG")
        handler.save(cutpcgtime,cutpcgval/6000,f"data/{savedname}PCG")
# print("refresh")



