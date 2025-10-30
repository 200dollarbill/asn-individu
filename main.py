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


# respiratory signal 0.15 - 0.4hz
p1 = st.Page("pages/page1.py", title="Respiratory Signal")
# respiratory rate
p2 = st.Page("pages/page2.py", title="Respiratory Rate Calculation")
# vasometric activity
p3 = st.Page("pages/page3.py", title="Vasometric Activity Signal")
# fft of vasometric activity & peak frequency
p4 = st.Page("pages/page4.py", title="DFT/FFT")


respPage = [p1,p2]
vasoPage = [p3,p4]


# nav page dictionary
mainNavigation = st.navigation({"Respiratory Signal Analysis": respPage, "Vasometric Activity Analysis":vasoPage})
mainNavigation.run()


