# Bank Reconciliation Benchmark - Learning Log

This document tracks the iterative discoveries, architectural insights, and prompt engineering lessons learned while attempting to push the TaskGraph Hybrid pipeline past 90% F1 on the "HARD" synthetic dataset.

## Learning 1: The "Whittling" Architecture (SQL -> LLM)
Pure LLM agents struggle with high-volume, exact data matching (our `strategy4_pure_prompt` scored ~79.5% F1). Conversely, pure SQL struggles with resolving ambiguous text or handling implicit tolerance bands (our `strategy3a_sql_only` scored 88.4% F1 and missed all wire fee mismatches). 

The hybrid TaskGraph approach proved effective by interleaving the two:
1. **SQL Nodes:** Deterministically resolve the straightforward matches. `match_certain`, `offsetting`, and `batch_match` resolved ~890 items (90% of the dataset) directly.
2. **LLM Node:** We handed the agent only the remaining ~100 ambiguous items. 

This architecture reduced token usage, mitigated context-window issues, and allowed the agent to focus on anomaly resolution.

## Learning 2: Immutability and Debuggability
TaskGraph's design outputs materialized views to a local DuckDB `.db` file (e.g. `strategy3_hybrid.db`). 
When the LLM failed to match "Weekly Recurring" payments, we connected to the `.db` file and queried the `_trace` table to see the exact SQL the LLM generated and the specific data pool it was evaluating (`SELECT * FROM match_exact_closest_final_remaining_bank`). 

Because the pipeline state is frozen at each node boundary, debugging agent behavior is a straightforward SQL querying exercise.

## Learning 3: Literal Data Tables (`VALUES`) for Edge Cases
**The Problem:** TaskGraph prompt nodes require the LLM to output a `CREATE VIEW` statement. For highly ambiguous 1-to-1 matches (like matching `AT&T *PAYMENT` to `Verizon Services` due to extreme truncation), forcing the LLM to write a generalized SQL `JOIN` rule can result in brittle logic and false positives.

**The Solution:** The `VALUES` clause is standard SQL. We added this instruction to the LLM prompt:
> *EXTREME OBFUSCATION (MANUAL MAPPING): When the candidate pool is very small and you see clear matches that SQL could never join... do NOT try to invent complex SQL join rules. Simply read the rows yourself, pair them up manually, and output a literal table using a `VALUES` clause.*

The LLM adopted this approach. Instead of attempting complex logic, it emitted literal tables:
```sql
SELECT * FROM (VALUES ('B012', 'G044', 100.50), ('B099', 'G102', 45.00)) AS t(bank_id, gl_id, amount);
```
This preserved TaskGraph's orthogonal SQL architecture while allowing the LLM to manually map the final fuzzy anomalies.

## Learning 4: Explicit Fuzzy Join Logic in Prompts
The `HARD` dataset includes typos in vendor names (e.g., `SKYLINE CRETIVE` instead of `SKYLINE CREATIVE`). 

In early iterations, the LLM only added ~2.8 points of F1 over the SQL-Only baseline. This occurred because the prompt explicitly instructed the LLM: `Use starts_with(gl_entity, bank_entity) for prefix matching.` The LLM followed this rule strictly and failed to match typo-laden batch deposits.

We updated the prompt to explicitly permit fuzzy SQL functions:
> *Use `starts_with()` OR `jaro_winkler_similarity() > 0.85` to account for typos in vendor names.*

**The Result:** The LLM updated its CTEs to include Jaro-Winkler thresholds. Batch match recall improved significantly, and the total F1 increased from 91.2% to 94.7%. This demonstrates that the LLM requires explicit permission in the prompt to utilize fuzzy matching functions when generating SQL.
