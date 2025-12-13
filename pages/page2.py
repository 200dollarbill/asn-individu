
import matplotlib
# matplotlib.use('module://matplotlib_kitty') # 
import matplotlib.pyplot as plt
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from deps import handler
from dwt_coeff import DWTCoeff


var = DWTCoeff()

scalecount = 8
FS = 50
delay_data = {
    "Skala j": [],
    "Delay index": [],
    "Delay dalam detik": []
}

for j in range(1, scalecount + 1):
    T = round(2**(j-1)) - 1
    delay_data["Skala j"].append(j)
    delay_data["Delay index"].append(T)
    delay_data["Delay dalam detik"].append(T / FS)

df_delay = pd.DataFrame(delay_data)
N_FFT = 2048 
coeff_generator = DWTCoeff()
g = coeff_generator.get_filter(scale=1) 

h = np.array([1, 3, 3, 1]) / 8.0 

Gw = np.abs(np.fft.rfft(g, N_FFT))
Hw = np.abs(np.fft.rfft(h, N_FFT))
freq_axis = np.fft.rfftfreq(N_FFT, 1/FS) 
Q = np.zeros((scalecount + 1, len(freq_axis)))

for j in range(1, scalecount + 1):
    temp_Q = np.interp(freq_axis, freq_axis / (2**(j-1)), Gw)
    for k in range(j - 1):
        temp_Hw = np.interp(freq_axis, freq_axis / (2**k), Hw)
        temp_Q *= temp_Hw
        
    Q[j] = temp_Q

# plt.style.use('seaborn-v0_8-whitegrid')
# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_alpha(0.0)
# ax.patch.set_alpha(0.0)

# for j in range(1, NUM_SCALES_TO_ANALYZE + 1):
#     ax.plot(freq_axis, Q[j], label=f"Skala {j}")

# ax.set_title('Frequency Response', color='white')
# ax.set_xlabel('Frekuensi (Hz)',color='white')
# ax.set_ylabel('Magnitude',color='white')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# ax.spines['left'].set_color('white')
# ax.spines['bottom'].set_color('white')
# ax.set_xlim(0, FS / 2) 
# ax.legend(labelcolor='white')
# st.pyplot(plt)



fig = go.Figure()

for j in range(1, scalecount + 1):
    fig.add_trace(go.Scatter(
        x=freq_axis, 
        y=Q[j], 
        mode='lines',
        name=f"Skala {j}" 
    ))

fig.update_layout(
    title='Frequency Response',
    xaxis_title='Frekuensi (Hz)',
    yaxis_title='Magnitude',
    xaxis_range=[0, FS / 2],       
    template='plotly_dark',       
    paper_bgcolor='rgba(0,0,0,0)',  
    plot_bgcolor='rgba(0,0,0,0)',   
    legend_title_text='Scales'     
)

st.plotly_chart(fig, use_container_width=True)

for j in range(1, scalecount + 1):
    fig.add_trace(go.Scatter(
        x=freq_axis, 
        y=Q[j], 
        mode='lines',
        name=f"Skala {j}" 
    ))

range_data = []
for j in range(1, scalecount + 1):
    f_min = FS / (2**(j + 1))
    f_max = FS / (2**j)
    bandwidth = f_max - f_min
    range_data.append({
        "Skala": j,
        "Frekuensi Minimum (Hz)": f_min,
        "Frekuensi Maksimum (Hz)": f_max,
        "Bandwidth (Hz)": bandwidth
    })

df_range = pd.DataFrame(range_data)

# st.dataframe(df_delay)
st.dataframe(df_range)
