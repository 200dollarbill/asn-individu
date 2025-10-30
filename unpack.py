import wfdb

record = wfdb.rdrecord('100') 

signal_data = record.p_signal
print(f"Signal data shape: {signal_data.shape}")
print(f"Sampling frequency: {record.fs} Hz")
print(f"Signal names: {record.sig_name}")
print(f"Signal units: {record.units}")
print(f"Record comments: {record.comments}")
wfdb.plot_wfdb(record=record, title='Record 100 from MIT-BIH',
               time_units='seconds', end_samp=5000)