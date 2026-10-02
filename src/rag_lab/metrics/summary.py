def summary_markdown(columns: list[str], rows: list[tuple]) -> str:
    """The experiment_summary view as a Markdown table, one column per experiment so that
    experiments can be read side by side. The first two columns are the name and config hash."""
    headers = [f"{r[0]} ({r[1][:6]})" for r in rows]  # name plus the start of the config hash
    lines = [
        "| metric | " + " | ".join(headers) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for i, column in enumerate(columns[2:], start=2):
        cells = ["" if r[i] is None else str(r[i]) for r in rows]
        lines.append(f"| {column} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
