import json
from datetime import datetime
from examples.bank_rec_v4 import NODES as V4_NODES

def load_bank():
    with open("dataset.json") as f:
        data = json.load(f)["bank_transactions"]
        for row in data:
            row["date"] = datetime.strptime(row["date"], "%Y-%m-%d").date()
        return data

def load_gl():
    with open("dataset.json") as f:
        data = json.load(f)["gl_entries"]
        for row in data:
            row["date"] = datetime.strptime(row["date"], "%Y-%m-%d").date()
        return data

NODES = [
    {"name": "bank_txns", "source": load_bank},
    {"name": "gl_entries", "source": load_gl},
] + [n for n in V4_NODES if n["name"] not in ("bank_txns", "gl_entries")]
