import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dwt_coeff import DWTCoeff  # Your custom module
from deps import handler        # Your custom module

# --- 1. Streamlit Page Configuration ---
# Set the page to use a wide layout for better plot visualization.
st.set_page_config(layout="wide")
st.title("PPG Signal Analysis with DWT")

# --- 2. Data Loading and Processing ---
# This section is identical to your original script.
# For better performance on repeated runs, we can cache the results of this heavy computation.
@st.cache_data
def compute_dwt_coefficients():
    """
    Loads data and computes the DWT coefficients.
    The @st.cache_data decorator stores the result so it doesn't need to be re-calculated
    every time you interact with the app, making it much faster.
    """
    coeff = DWTCoeff()
    var = handler.load("rawdata")
    ppgdata = var.value.to_numpy()
    time = var.time.to_numpy()
    total = len(ppgdata)

    w2fb = np.zeros((9, total))
    scalecount = 8

    # a trous algo
    for j in range(1, scalecount + 1):
        res = coeff.get_filter(scale=j)
        T = round(2**(j-1)) - 1
        start = len(res)
        for n in range(start, total):
            signalNEW = ppgdata[n - len(res):n]
            w2fb[j, n - T] = np.sum(signalNEW * res[::-1])
    
    return time, ppgdata, w2fb, scalecount

# Execute the function to get the data and coefficients
try:
    time, ppgdata, w2fb, scalecount = compute_dwt_coefficients()

    st.success("Data loaded and DWT coefficients computed successfully.")

    # --- 3. Plotting Section ---
    # This section iterates through the scales and creates a plot for each one.
    # Instead of fig.show(), we use st.plotly_chart() to display the plots in Streamlit.

    st.header("DWT Coefficients at Different Scales")

    for j in range(1, scalecount + 1):
        fig = go.Figure()

        # Add the DWT coefficient trace
        fig.add_trace(go.Scatter(
            x=time,
            y=w2fb[j],
            mode='lines',
            line=dict(color='blue'),
            name=f'DWT Skala {j}'
        ))

        # Add the original PPG signal for comparison
        fig.add_trace(go.Scatter(
            x=time,
            y=ppgdata,
            mode='lines',
            line=dict(color='red'),
            name='PPG Baseline',
            opacity=0.4
        ))

        # Update layout for titles and labels
        fig.update_layout(
            title=f"Hasil DWT Skala {j}",
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            height=400,
            legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
            margin=dict(t=50, b=40, l=40, r=20)
        )

        # Use Streamlit to display the Plotly figure
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred during data loading or processing.")
    st.error(f"Please ensure your 'dwt_coeff.py', 'deps.py', and raw data are correctly set up.")
    st.exception(e)