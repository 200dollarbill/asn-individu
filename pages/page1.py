import streamlit as st
import numpy as np
from deps import handler
from pan_tompkins import Pan_Tompkins_QRS, HeartRate

fs = 2000 
ecgraw = handler.load("data/ecg")
ecgtime = ecgraw.time
ecgval = ecgraw.value

pcgraw = handler.load("data/pcg")
pcgval = pcgraw.value
pcgtime = pcgraw.time

# type for each = ndarray
# shape = (71193,)
# st.write(ecgtime.shape)

# downsample

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


