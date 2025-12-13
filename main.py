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


# DWT inputs scale 1 - 8
p1 = st.Page("pages/page1.py", title="Respiratory Signal DWT Filtering")
# DWT Frequency Response
p2 = st.Page("pages/page2.py", title="DWT Filter Frequency Response")
# vasometric activity
p3 = st.Page("pages/page3.py", title="Breath Rate Calculation")
# breath rate tachogram
p4 = st.Page("pages/page4.py", title="Breath Rate Tachogram")
# heart rate analysis
p5 = st.Page("pages/page5.py", title="Heart Rate Analysis")
# RR F domain analysis
p6 = st.Page("pages/page6.py", title="RR Frequency Analysis")




respPage = [p1,p2,p3,p4]
RRPage = [ p5, p6]


# nav page dictionary
mainNavigation = st.navigation({"Respiratory Signal Analysis": respPage, "RR Analysis":RRPage})
mainNavigation.run()


