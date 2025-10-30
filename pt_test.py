import numpy as np
SAMPLING_FREQUENCY = 360 
duration = 10 # seconds
num_samples = int(duration * SAMPLING_FREQUENCY)

# Simulate a simple ECG signal
time_ecg = np.arange(num_samples)
# A very basic simulation of R-peaks every second (60 BPM)
r_peak_indices = np.arange(SAMPLING_FREQUENCY, num_samples, SAMPLING_FREQUENCY)
ecg_signal = np.zeros(num_samples)
for idx in r_peak_indices:
    # Create a sharp peak
    ecg_signal[idx-2:idx+3] = [0.5, 2.0, 5.0, 2.0, 0.5]
# Add some noise
ecg_signal += np.random.normal(0, 0.2, num_samples)

# This simulates the data structure you described
class ECG:
    def __init__(self, values, time_indices):
        self.value = values
        self.time = time_indices

ecg = ECG(ecg_signal, time_ecg)
# --- End of sample data creation ---


# 1. Import the classes from your new module
from pan_tompkins import Pan_Tompkins_QRS, HeartRate

# 2. Prepare your data and parameters
signal_values = ecg.value
fs = SAMPLING_FREQUENCY 
qrs_detector = Pan_Tompkins_QRS()
bpass_signal, _, _, mwin_signal = qrs_detector.solve(signal_values, fs)

hr_detector = HeartRate(raw_signal=signal_values, mwin_signal=mwin_signal, bpass_signal=bpass_signal, samp_freq=fs)

# 5. Find the R-peaks
r_peaks = hr_detector.find_r_peaks()

# 6. Post-process and calculate heart rate
# Clip the R-peaks found during the learning phase (first few seconds)
r_peaks = r_peaks[r_peaks > (0.5 * fs)] # Ignore peaks in the first 0.5 seconds

print(f"Detected R-peak locations (indices): {r_peaks}")

# Calculate the average RR interval in samples
rr_intervals = np.diff(r_peaks)
avg_rr_interval = np.mean(rr_intervals)

# Calculate heart rate in Beats Per Minute (BPM)
heart_rate_bpm = (fs * 60) / avg_rr_interval
print(f"\nCalculated Heart Rate: {heart_rate_bpm:.2f} BPM")
