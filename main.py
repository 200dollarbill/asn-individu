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


# setup page

p1 = st.Page("pages/page1.py", title="Data Loading")
p2 = st.Page("pages/page2.py", title="DWT")


dataLoadingPage = [p1, p2]

# nav page dictionary
mainNavigation = st.navigation({"Breath Rate": dataLoadingPage})
mainNavigation.run()


