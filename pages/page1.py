import mne 
import pandas as pd

import streamlit as st
raw = mne.io.read_raw_gdf(input_fname="../dataset/TRAINING/B0101T.gdf")
events, event_id = mne.events_from_annotations(raw)

mapping = {v: k for k, v in event_id.items() if '769' in k or '770' in k}
label_map = {v: ("Left Hand" if '769' in k else "Right Hand") for k, v in event_id.items() if '769' in k or '770' in k}

event_df = pd.DataFrame(events, columns=['sample_index', 'zeros', 'event_id'])
event_df['label'] = event_df['event_id'].map(label_map)

trial_df = event_df.dropna(subset=['label'])
st.write("### Trial Labels and Locations", trial_df)

st.write("### Found Events:", label_map)

rawdf = raw.to_data_frame()
st.write(rawdf)
time = rawdf['time']
c3 = rawdf['EEG:C3']
cz = rawdf['EEG:Cz']
c4 = rawdf['EEG:C4']
print(rawdf)

st.write("### Evaluation Data")
print("trainig data")
raw1 = mne.io.read_raw_gdf(input_fname="../dataset/EVALUATION/B0104E.gdf")
st.write(raw1)
rawdf1 = raw1.to_data_frame()

events1, event_id1 = mne.events_from_annotations(raw1)
mapping1 = {v: k for k, v in event_id1.items() if '769' in k or '770' in k}
label_map1 = {v: ("Left Hand" if '769' in k else "Right Hand" if '770' in k else "Unknown/Cue") 
                 for k, v in event_id1.items()}
event_df1 = pd.DataFrame(events1, columns=['sample_index', 'zeros', 'event_id'])
event_df1['label'] = event_df1['event_id'].map(label_map1)

trial_df = event_df1.dropna(subset=['label'])
st.write("### Trial Labels and Locations", trial_df)

st.write("### Found Events:", label_map1)
st.write(rawdf1)
# st.dataframe(rawdf1)
time = rawdf1['time']
c3 = rawdf1['EEG:C3']
cz = rawdf1['EEG:Cz']
c4 = rawdf1['EEG:C4']
print(rawdf1)