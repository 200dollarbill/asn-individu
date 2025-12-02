import streamlit as st

page_dict = {}

firstpage = st.Page("pages/page1.py", title="Data loading")
secondpage = st.Page("pages/page2.py", title="")
thirdpage = st.Page("pages/page3.py", title="")
dataLoadingPage = [firstpage, secondpage, thirdpage]

# nav page dictionary
mainNavigation = st.navigation({"Load Data": dataLoadingPage})

mainNavigation.run()


