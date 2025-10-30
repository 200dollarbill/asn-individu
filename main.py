import streamlit as st

page_dict = {}
# if "page" not in st.session_state:
#     st.session_state.page = None

# active sessions
#setup sessions by buttons

# st.button("button1")
# if st.button("button1"):
#     st.session_state.page = pages[0]
#     st.rerun()

# st.button("button2")


# if st.button("button2"):
#     st.session_state.page = pages[1]
#     st.rerun()

# activepage = st.session_state.page 


# pan tompkins algorithm for rr interval finding
p1 = st.Page("pages/page1.py", title="Pan Tompkins algorithm application")
# Segmenting and lag adjusment
p2 = st.Page("pages/page2.py", title="Segmenting")
# CWT analysis
p3 = st.Page("pages/page3.py", title="Vasometric Activity Signal")
# magnitude thresholding and COG caluclation
p4 = st.Page("pages/page4.py", title="DFT/FFT")


TDPage = [p1,p2]
CWTPage = [p3,p4]


# nav page dictionary
mainNavigation = st.navigation({"Time Domain Analysis": TDPage, "Time-Frequency Domain Analysis":CWTPage})
mainNavigation.run()


