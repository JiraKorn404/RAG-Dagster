import os
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st
import style

from rag_lab.config import ImportConfig, SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.sql import catalog, importer
from rag_lab.sql import examples as good_answers
from rag_lab.sql.database import create_schema, drop_schema, drop_table
from rag_lab.storage.qdrant import QdrantStore

CFG = ImportConfig()
DELIMITER_CHOICES = {"detect": None, "comma ,": ",", "semicolon ;": ";", "tab": "\t", "bar |": "|"}
PROBLEMS = (ValueError, RuntimeError, psycopg.Error)  # shown to the user, not raised (a CsvError is a ValueError)


@st.cache_resource
def store() -> MetricsStore:
    return MetricsStore(os.environ["METRICS_DATABASE_URL"])


@st.cache_resource
def example_services() -> tuple[OllamaEmbedder, QdrantStore]:
    return OllamaEmbedder(os.environ["OLLAMA_BASE_URL"]), QdrantStore(os.environ["QDRANT_URL"])


@st.cache_data(max_entries=8, show_spinner=False)
def sample_of(file_id: str, delimiter: str | None, _data: bytes) -> importer.CsvSample:
    """The sample of an uploaded file, read once for an upload and a delimiter, not on every rerun."""
    return importer.parse_sample(_data, delimiter, CFG)


def import_panel(file, schema: str, tables: dict[str, dict]) -> None:
    """One uploaded file: what it looks like, what it will become, and the button that imports it."""
    data = file.getvalue()
    key = f"{schema}-{file.name}-{file.size}"
    with st.container(border=True):
        st.markdown(f"**{file.name}** · {len(data) / 1e6:.1f} MB")
        if len(data) > CFG.max_bytes:
            st.error(f"The limit is {CFG.max_bytes / 1e6:.0f} MB per file.")
            return
        c1, c2 = st.columns([1, 2])
        choice = c1.selectbox("Delimiter", list(DELIMITER_CHOICES), key=f"delimiter-{key}")
        try:
            sample = sample_of(file.file_id, DELIMITER_CHOICES[choice], data)
        except importer.CsvError as e:
            st.error(str(e))
            return
        table = c2.text_input(
            "Table name",
            value=importer.clean_identifier(Path(file.name).stem, "table"),
            key=f"table-{key}",
            help="Lower case letters, digits and _. Anything else is cleaned.",
        )
        table = importer.clean_identifier(table, "table")
        st.caption(
            f"Delimiter: {'tab' if sample.delimiter == chr(9) else sample.delimiter}. The first row is the header. "
            f"The types below are guessed from the first {len(sample.rows):,} rows"
            + (" of a longer file." if sample.truncated else ".")
        )
        st.dataframe(pd.DataFrame(sample.rows[: CFG.preview_rows], columns=sample.headers), hide_index=True)
        columns = pd.DataFrame({"column": sample.names, "header": sample.headers, "type": sample.types})
        edited = st.data_editor(
            columns,
            key=f"types-{key}",
            hide_index=True,
            disabled=["column", "header"],
            column_config={"type": st.column_config.SelectboxColumn("type", options=list(importer.TYPES), required=True)},
        )
        exists = table in tables
        replace = False
        if exists:
            replace = st.checkbox(
                f"The table `{table}` exists ({tables[table]['row_count']:,} rows). Replace it.", key=f"replace-{key}"
            )
        if st.button(f"Import into {schema}.{table}", key=f"import-{key}", type="primary", disabled=exists and not replace):
            try:
                with st.spinner("Importing…"):
                    rows = importer.import_csv(
                        store(),
                        schema,
                        table,
                        data,
                        delimiter=sample.delimiter,
                        headers=sample.headers,
                        names=sample.names,
                        types=list(edited["type"]),
                        replace=replace,
                        source_file=file.name,
                        cfg=CFG,
                    )
            except PROBLEMS as e:
                st.error(f"Not imported: {e}")
            else:
                st.success(f"Imported {rows:,} rows into {schema}.{table}.")
                tables[table] = {"row_count": rows}


def show_tables(schema: str) -> None:
    tables = store().list_db_tables(schema)
    style.section(
        f"Tables in {schema}",
        "What was imported, and what each column holds. Descriptions are optional and are shown to the model with the table.",
    )
    if not tables:
        st.caption("No tables yet.")
    for t in tables:
        name = t["table_name"]
        with st.expander(f"{name} · {t['row_count']:,} rows · {len(t['columns'])} columns"):
            st.caption(f"From {t['source_file'] or 'a file'}, imported {t['imported_at']:%d %b %Y %H:%M} UTC")
            description = st.text_input(
                "What this table holds",
                value=t["description"],
                key=f"table-description-{schema}-{name}",
                placeholder="One row per employee",
            )
            profile = pd.DataFrame(t["columns"]).reindex(
                columns=["name", "type", "description", "header", "nulls", "distinct", "samples", "min", "max"]
            )
            profile["samples"] = profile["samples"].map(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else "")
            edited = st.data_editor(
                profile,
                key=f"columns-{schema}-{name}",
                hide_index=True,
                disabled=[c for c in profile.columns if c != "description"],
                column_config={"description": st.column_config.TextColumn("description", help="Write what the column means, and what odd values mean")},
            )
            if st.button("Save descriptions", key=f"save-{schema}-{name}"):
                try:
                    store().set_db_table_descriptions(
                        schema,
                        name,
                        description.strip(),
                        {c: (d or "").strip() for c, d in zip(edited["name"], edited["description"])},
                    )
                except PROBLEMS as e:
                    st.error(f"Not saved: {e}")
                else:
                    st.toast(f"Saved the descriptions of {name}")
                    st.rerun()
            with st.popover("Delete this table"):
                st.write("This drops the table and its saved CSV.")
                if st.button("Yes, delete it", key=f"drop-{schema}-{name}"):
                    try:
                        drop_table(store(), schema, name)
                    except PROBLEMS as e:
                        st.error(f"Not deleted: {e}")
                    else:
                        st.rerun()


def show_schema_text(schema: str, current: dict) -> None:
    """What the text-to-SQL agent is given about the schema, and how much of its budget that is."""
    budget = SqlAgentConfig().schema_char_budget
    style.section("What the model sees", "The whole schema goes in front of the model with every question, so keep it short.")
    description = st.text_input(
        "What this dataset is", value=current["description"], key=f"schema-description-{schema}", placeholder="HR data of the company"
    )
    if st.button("Save", key=f"save-schema-{schema}"):
        store().set_db_schema_description(schema, description.strip())
        st.toast("Saved")
        st.rerun()
    text = catalog.render(store(), schema)
    st.progress(min(text.chars / budget, 1.0), text=f"{text.chars:,} of {budget:,} characters · {len(text.tables)} table(s)")
    try:
        text.check(budget)
    except catalog.SchemaTooLarge as e:
        st.error(str(e))
    with st.expander("Show the text"):
        st.code(text.text, language="text")


def show_examples(schema: str) -> None:
    """The good answers saved for a schema: what the model is shown when a similar question comes."""
    found = store().list_sql_examples(schema)
    style.section(
        "Good answers",
        "Questions marked as a good answer in the Chatbot, with the SQL that answered them. The model is shown the ones "
        "most like a new question. Turn one off to stop it being used, or delete it.",
    )
    if not found:
        st.caption("None yet. In a database chat, click *Good answer* under an answer that is right.")
    embedder, qdrant = example_services()
    for e in found:
        with st.expander(("" if e["enabled"] else "(off) ") + e["standalone"]):
            st.code(e["sql"], language="sql")
            if e["question"] != e["standalone"]:
                st.caption(f"Asked as: {e['question']}")
            st.caption(f"Saved {e['created_at']:%d %b %Y %H:%M} UTC")
            c1, c2 = st.columns(2)
            try:
                if c1.button("Turn off" if e["enabled"] else "Turn on", key=f"toggle-example-{e['id']}"):
                    good_answers.set_enabled(store(), embedder, qdrant, SqlAgentConfig(), schema, e, not e["enabled"])
                    st.rerun()
                if c2.button("Delete", key=f"delete-example-{e['id']}"):
                    good_answers.remove(store(), qdrant, schema, e["id"])
                    st.rerun()
            except Exception as err:  # noqa: BLE001  (Ollama or Qdrant down: the table may be ahead of the index)
                st.error(f"Not changed: {err}. `python -m rag_lab.sql examples --reindex` rebuilds the index from the table.")


style.hero("Database", "Import CSV files into a schema, one dataset per schema")

schemas = store().list_db_schemas()
with st.expander("New schema", expanded=not schemas):
    with st.form("new-schema", clear_on_submit=True):
        new_name = st.text_input("Schema name", placeholder="sales", help="Lower case letters, digits, _ and -.")
        description = st.text_input("Description (optional)")
        if st.form_submit_button("Create schema"):
            try:
                create_schema(store(), new_name.strip(), description.strip())
            except PROBLEMS as e:
                st.error(f"Not created: {e}")
            else:
                st.session_state["db_schema"] = new_name.strip()
                st.rerun()

if not schemas:
    st.info("Make a schema to import CSV files into.")
    st.stop()

names = [s["name"] for s in schemas]
if st.session_state.get("db_schema") not in names:
    st.session_state.pop("db_schema", None)
by_name = {s["name"]: s for s in schemas}
schema = st.selectbox("Schema", names, key="db_schema", format_func=lambda n: f"{n} · {by_name[n]['tables']} table(s)")

style.section("Import CSV files", f"Up to {CFG.max_bytes / 1e6:.0f} MB and {CFG.max_rows:,} rows each, UTF-8, first row the header.")
files = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True, key=f"csv-{schema}")
existing = {t["table_name"]: t for t in store().list_db_tables(schema)}
for file in files:
    import_panel(file, schema, existing)

show_tables(schema)
show_schema_text(schema, by_name[schema])
show_examples(schema)

with st.popover("Delete this schema"):
    st.write(f"This deletes `{schema}`, all its tables and their saved CSV files, and the chats that searched it.")
    if st.button("Yes, delete the schema", key=f"drop-schema-{schema}"):
        try:
            drop_schema(store(), schema, example_services()[1])
        except PROBLEMS as e:
            st.error(f"Not deleted: {e}")
        else:
            st.rerun()
