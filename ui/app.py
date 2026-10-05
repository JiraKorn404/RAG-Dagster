import os

import streamlit as st
import style
from rag_lab.metrics.migrate import apply_migrations
from rag_lab.sql.bootstrap import ensure_database

st.set_page_config(page_title="RAG Lab", page_icon=":material/hub:", layout="wide")
style.inject()


@st.cache_resource
def _migrate() -> None:
    # The schema is otherwise only brought up to date by a Dagster run.
    apply_migrations(os.environ["METRICS_DATABASE_URL"])


_migrate()


@st.cache_resource
def _prepare_sql() -> None:
    # The database for imported tables, its roles and their limits. A failure is not cached, so it is
    # tried again on the next run.
    ensure_database()


try:
    _prepare_sql()
except Exception as e:  # noqa: BLE001  (the other pages do not need it: say what is wrong and carry on)
    st.warning(f"The database for imported tables is not ready: {e}")

pages = [
    st.Page("experiments.py", title="Experiments", icon=":material/folder_managed:"),
    st.Page("upload.py", title="Upload", icon=":material/upload_file:", default=True),
    st.Page("query.py", title="Try a query", icon=":material/search:"),
    st.Page("chat.py", title="Chatbot", icon=":material/chat:"),
    st.Page("database.py", title="Database", icon=":material/database:"),
    st.Page("benchmark.py", title="Benchmark", icon=":material/speed:"),
]
st.navigation(pages).run()
