"""CLI: python -m rag_lab.agent documents ["question"] --experiment <name> [--top-k 5] [--model <name>] [--no-think]
      python -m rag_lab.agent sql ["question"] --schema <name> [--model <name>] [--no-think]

One subcommand for each flow. `documents` answers from the documents in an experiment, `sql` from the
tables of an imported schema."""

import argparse

from rag_lab.agent.documents import cli as documents
from rag_lab.agent.sql import cli as sql


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab.agent", description=__doc__.split("\n")[0])
    flows = parser.add_subparsers(dest="flow", required=True, metavar="flow")
    documents.add_parser(flows)
    sql.add_parser(flows)
    args = parser.parse_args()
    args.main(args)


if __name__ == "__main__":
    main()
