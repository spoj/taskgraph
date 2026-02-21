import json
from datetime import datetime
from examples.bank_rec.strategy3_hybrid import NODES as HYBRID_NODES


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


REPORT_SQL = """
CREATE VIEW report_matched AS
SELECT * FROM match_certain_matched
UNION ALL
SELECT * FROM batch_match_all_batch_matched
UNION ALL
SELECT * FROM match_exact_closest_matched;

CREATE VIEW report_unmatched_bank AS
SELECT b.*
FROM features_bank b
JOIN match_exact_closest_final_remaining_bank r ON b.id = r.id;

CREATE VIEW report_unmatched_gl AS
WITH offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.*
FROM features_gl g
JOIN match_exact_closest_final_remaining_gl r ON g.id = r.id
LEFT JOIN offset_gl o ON g.id = o.id
WHERE o.id IS NULL;

CREATE VIEW report_summary AS
SELECT
    (SELECT count(*) FROM match_certain_matched) AS certain_matches,
    (SELECT count(*) FROM batch_match_all_batch_matched) AS batch_matches,
    (SELECT count(*) FROM match_exact_closest_matched) AS exact_closest_matches,
    (SELECT count(*) FROM report_unmatched_bank) AS unmatched_bank,
    (SELECT count(*) FROM report_unmatched_gl) AS unmatched_gl;
"""

sql_nodes = [
    dict(n)
    for n in HYBRID_NODES
    if n["name"] not in ("bank_txns", "gl_entries", "match_residual", "report")
]
sql_nodes.append(
    {
        "name": "report",
        "sql": REPORT_SQL,
        "depends_on": ["features", "match_exact_closest", "batch_match"],
    }
)

NODES = [
    {"name": "bank_txns", "source": load_bank},
    {"name": "gl_entries", "source": load_gl},
] + sql_nodes
