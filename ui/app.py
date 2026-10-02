import streamlit as st
import style

st.set_page_config(page_title="RAG Lab", page_icon=":material/hub:", layout="wide")
style.inject()

pages = [
    st.Page("dashboard.py", title="Dashboard", icon=":material/monitoring:", default=True),
    st.Page("query.py", title="Try a query", icon=":material/search:"),
]
st.navigation(pages).run()
