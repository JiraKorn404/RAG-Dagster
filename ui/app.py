import streamlit as st
import style

st.set_page_config(page_title="RAG Lab", page_icon=":material/hub:", layout="wide")
style.inject()

pages = [
    st.Page("experiments.py", title="Experiments", icon=":material/folder_managed:"),
    st.Page("upload.py", title="Upload", icon=":material/upload_file:", default=True),
    st.Page("query.py", title="Try a query", icon=":material/search:"),
    st.Page("benchmark.py", title="Benchmark", icon=":material/speed:"),
]
st.navigation(pages).run()
