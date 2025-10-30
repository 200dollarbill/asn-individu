import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dwt_coeff import DWTCoeff  
from deps import handler        

st.set_page_config(layout="wide")
st.title("PPG Signal Analysis with DWT")

@st.cache_data
def compute_dwt_coefficients():
    coeff = DWTCoeff()
    var = handler.load("rawdata")
    ppgdata = var.value.to_numpy()
    time = var.time.to_numpy()
    total = len(ppgdata)
    w2fb = np.zeros((9, total))
    scalecount = 8
    for j in range(1, scalecount + 1):
        res = coeff.get_filter(scale=j)
        T = round(2**(j-1)) - 1
        start = len(res)
        for n in range(start, total):
            signalNEW = ppgdata[n - len(res):n]
            w2fb[j, n - T] = np.sum(signalNEW * res[::-1])
    
    return time, ppgdata, w2fb, scalecount

time, ppgdata, w2fb, scalecount = compute_dwt_coefficients()

for j in range(1, scalecount + 1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=w2fb[j],
        mode='lines',
        line=dict(color='blue'),
        name=f'DWT Skala {j}'
    ))

    fig.add_trace(go.Scatter(
        x=time,
        y=ppgdata,
        mode='lines',
        line=dict(color='red'),
        name='PPG Baseline',
        opacity=0.4
    ))

    fig.update_layout(
        title=f"Hasil DWT Skala {j}",
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=400,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
        margin=dict(t=50, b=40, l=40, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

handler.save(time,w2fb[7], filename="dwt8")