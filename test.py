import wfdb
import streamlit as st
import numpy as np
record = wfdb.rdrecord('data/S24')
st.write(record.sig_name)
st.write(record)
# st.write(record.p_signal)
RT_FOOT = record.p_signal[:,7]
LAT_G = record.p_signal[:,10]
LAT_V = record.p_signal[:,13]

time = np.arange(len(LAT_G))


st.write(time)