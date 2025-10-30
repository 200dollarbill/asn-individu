import wfdb
import numpy as np
from deps import handler


record = wfdb.rdrecord('data/a0004') 

pcgY = record.p_signal[:,0]
ecgY = record.p_signal[:,1]
pcgX = np.arange(record.p_signal.shape[0])
ecgX = np.arange(record.p_signal.shape[0])
print(ecgX)
print(pcgX)

signal_data = record.p_signal
print(f"Signal data shape: {signal_data.shape}")
print(f"Sampling frequency: {record.fs} Hz")
print(f"Signal names: {record.sig_name}")
print(f"Signal units: {record.units}")
print(f"Record comments: {record.comments}")
handler.save(ecgX,ecgY,filename="ecg")
handler.save(pcgX,pcgY,filename="pcg")

wfdb.plot_wfdb(record=record,time_units='seconds')



