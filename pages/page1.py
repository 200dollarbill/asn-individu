import streamlit as st
import numpy as np
from deps import handler
# from pan_tompkins import Pan_Tompkins_QRS, HeartRate
import plotly.graph_objects as go
import wfdb


foot_plot = go.Figure()

pcgY=np.arange(1)
cutecgval= np.arange(1)
cutecgtime=np.arange(1)
cutpcgval= np.arange(1)
cutpcgtime=np.arange(1)
foot_plot = go.Figure()
latg_plot = go.Figure()
latv_plot = go.Figure()

st.set_page_config(layout="wide")
if 'show_threshold' not in st.session_state:
    st.session_state.show_threshold = False
if 'show_input' not in st.session_state:
    st.session_state.show_input = False
if 'thresholded' not in st.session_state:
    st.session_state.thresholded = False

with st.form(key='input'):
    name = st.text_input("Input Data")
    input = st.form_submit_button(label="Load Data")
    clear = st.form_submit_button(label="Clear")

lower_sec_limit = st.number_input(label="Lower Limit (seconds)", value=0,key="LOWER")
upper_sec_limit = st.number_input(label="Upper Limit (seconds)", value=10,key="UPPER")
# st.number_input()


if int(upper_sec_limit) == 0:
    upper_sec_limit == 0

if int(lower_sec_limit) == 0:
    lower_sec_limit == 10


if input or st.session_state.show_input:
    name = str(name)
    st.write(upper_sec_limit)
    record = wfdb.rdrecord('data/' + name)
    fs = record.fs
    upper_limit = int(upper_sec_limit * fs)   
    lower_limit = int(lower_sec_limit * fs)  
    RT_FOOT = record.p_signal[lower_limit:upper_limit,7]
    LAT_G = record.p_signal[lower_limit:upper_limit,10]
    LAT_V = record.p_signal[lower_limit:upper_limit,13]
    INDEX = np.arange(len(LAT_G))

    handler.save(INDEX,RT_FOOT,"footswitch")
    handler.save(INDEX,LAT_G,"g_lateralis")
    handler.save(INDEX,LAT_V, "v_lateralis")

       
    # st.write(f"Signal data name : {name}.dat")
    # st.write(f"Sampling frequency: {record.fs} Hz")
    # st.write(f"Signal names: {record.sig_name}")
    # st.write(f"Signal units: {record.units}")
    # st.write(f"Record comments: {record.comments}")

    foot_plot.add_trace(go.Scatter(
        x=INDEX, 
        y=RT_FOOT, 
        mode='lines',
        name="Skala"
    ))
    foot_plot.update_traces(line_color='#00A9FF')
    foot_plot.update_layout(
        xaxis_title="Time",
        yaxis_title="Data",
        title="RT_FOOT"
    )
    st.plotly_chart(foot_plot, use_container_width=True)

    latg_plot.add_trace(go.Scatter(
        x=INDEX, 
        y=LAT_G, 
        mode='lines',
        name="Skala"
    ))
    latg_plot.update_traces(line_color='#00A9FF')
    latg_plot.update_layout(
        xaxis_title="Time",
        yaxis_title="Data",
        title="LAT_G"
    )
    st.plotly_chart(latg_plot, use_container_width=True)
    
    latv_plot.add_trace(go.Scatter(
        x=INDEX, 
        y=LAT_V, 
        mode='lines',
        name="Skala"
    ))
    latv_plot.update_traces(line_color='#00A9FF')
    latv_plot.update_layout(
        xaxis_title="Time",
        yaxis_title="Data",
        title="LAT_V"
    )
    st.plotly_chart(latv_plot, use_container_width=True)

    


