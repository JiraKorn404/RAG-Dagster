import os

import streamlit as st
import style
from rag_lab.metrics.migrate import apply_migrations

st.set_page_config(page_title="RAG Lab", page_icon=":material/hub:", layout="wide")
style.inject()


@st.cache_resource
def _migrate() -> None:
    # The schema is otherwise only brought up to date by a Dagster run.
    apply_migrations(os.environ["METRICS_DATABASE_URL"])


_migrate()

pages = [
    st.Page("experiments.py", title="Experiments", icon=":material/folder_managed:"),
    st.Page("upload.py", title="Upload", icon=":material/upload_file:", default=True),
    st.Page("query.py", title="Try a query", icon=":material/search:"),
    st.Page("chat.py", title="Chatbot", icon=":material/chat:"),
    st.Page("benchmark.py", title="Benchmark", icon=":material/speed:"),
]
st.navigation(pages).run()
