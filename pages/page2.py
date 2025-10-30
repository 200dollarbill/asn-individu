import streamlit as st
import numpy as np
from deps import handler
import plotly.graph_objects as go


fig = go.Figure()


st.set_page_config(layout="wide")
if 'show_cwt' not in st.session_state:
    st.session_state.show_cwt = False
if 'show_input' not in st.session_state:
    st.session_state.show_input = False

with st.form(key='input'):
    name = st.text_input("Input Thresholded Data Name")
    input = st.form_submit_button(label="Load Data")
    clear = st.form_submit_button(label="Clear")
if input or st.session_state.show_input:
    pcg=handler.load(f"data/{name}PCG")
    ecg=handler.load(f"data/{name}ECG")
    

    pcgtime = pcg.time
    # st.write(len(pcgtime))
    len = len(pcgtime)
    pcgval = pcg.value
    ecgtime = ecg.time
    ecgval = ecg.value
    
    fig.data=[]
    fig.add_trace(go.Scatter(x=pcgtime, y=pcgval, mode='lines', line=dict(color='rgba(255, 176, 0, 0.8)', width=0.8)))
    fig.add_trace(go.Scatter(x=ecgtime, y=ecgval, mode='lines', line=dict(color='rgba(33, 185, 33, 0.8)', width=0.8)))
    fig.update_layout(title='Thresholded result',height=500, width=1200)
    
    st.plotly_chart(fig)

    st.write("work")
if clear:
    st.session_state.show_cwt = False
    st.session_state.show_input = False
    st.rerun()


