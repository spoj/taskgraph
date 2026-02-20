# Bank Reconciliation Benchmark - Learning Log

This document tracks the iterative discoveries, architectural insights, and prompt engineering lessons learned while attempting to push the TaskGraph Hybrid pipeline past 90% F1 on the "HARD" synthetic dataset.

## Learning 1: The "Whittling" Architecture (SQL -> LLM)
Pure LLM agents fail completely at high-volume, exact data matching (our `strategy4_pure_prompt` choked on 1000 records, scoring ~79% F1). Conversely, Pure SQL fails completely at resolving ambiguous, cryptic text or handling implicit tolerance bands (our `strategy3a_sql_only` scored 88.4% F1 and missed every single wire fee mismatch). 

The hybrid TaskGraph approach proved its worth by seamlessly interleaving the two:
1. **SQL Nodes:** Deterministically clear the board. `match_certain`, `offsetting`, and `batch_match` cleanly resolved ~890 items (90% of the dataset) instantly.
2. **LLM Node:** We handed the agent *only* the remaining ~100 messy, ambiguous items. 

This architecture drastically reduced token costs, prevented LLM context-window hallucinations, and allowed the agent to focus purely on cognitive anomaly resolution.

## Learning 2: Immutability and Infinite Debuggability
TaskGraph's design forces every node to output materialized views to a local DuckDB `.db` file (`runs/strategy3_hybrid.db`). 
When an LLM failed to match "Weekly Recurring" payments, we didn't have to guess why based on fleeting `stdout` logs. We simply connected to the `.db` file and queried the `_trace` table to see the exact SQL the LLM generated and the exact data pool it was looking at (`SELECT * FROM match_exact_closest_final_remaining_bank`). 

Because the pipeline state is frozen at every node boundary, debugging data engineering agents becomes a standard SQL querying exercise rather than a frustrating console-hunting exercise.

## Learning 3: Literal Data Tables (`VALUES`) for Edge Cases
**The Problem:** TaskGraph prompt nodes force the LLM to output a `CREATE VIEW` statement. For highly ambiguous 1-to-1 matches (like matching `AT&T *PAYMENT` to `Verizon Services` due to a generator error or extreme truncation), forcing the LLM to write a generalized SQL `JOIN` rule to pair them is mathematically brittle and prone to catastrophic false-positives.

**The Solution:** We recognized that the `VALUES` clause is standard SQL. We added this rule to the LLM prompt:
> *EXTREME OBFUSCATION (MANUAL MAPPING): When the candidate pool is very small and you see clear matches that SQL could never join... do NOT try to invent complex SQL join rules. Simply read the rows yourself, pair them up manually, and output a literal table using a `VALUES` clause.*

The LLM brilliantly adopted this. Instead of writing broken `CASE WHEN` logic, it emitted literal tables:
```sql
SELECT * FROM (VALUES ('B012', 'G044', 100.50), ('B099', 'G102', 45.00)) AS t(bank_id, gl_id, amount);
```
This perfectly preserved TaskGraph's orthogonal SQL architecture while allowing the LLM to act as a "Data Annotator" for the final few fuzzy anomalies.

## Learning 4: Fuzzy Join Logic in Prompts is King
The `HARD` dataset injects chaotic typos into vendor names (e.g. `SKYLINE CRETIVE` instead of `SKYLINE CREATIVE`). 

In our early iterations, the LLM was only adding ~2.8 points of F1 over the SQL-Only ablation baseline. Why? Because the prompt instructed the LLM: `Use starts_with(gl_entity, bank_entity) for prefix matching.` The LLM rigidly followed this and failed to match all typo-laden batch deposits.

We updated the prompt to explicitly permit fuzzy SQL functions:
> *Use `starts_with()` OR `jaro_winkler_similarity() > 0.85` to account for typos in vendor names.*

**The Impact:** The LLM instantly rewrote its CTEs to include Jaro-Winkler thresholds. Batch match recall skyrocketed, and the total F1 jumped from **91.2% to 94.7%**. This highlights the immense power of TaskGraph: you get the execution speed of SQL joined with the contextual adaptability of an LLM. 
