# eval_v2: family-expanded silver evaluation

`eval_v2` supplements, but never replaces, the manually reviewed `eval_v1`.
It contains three newly generated variants for each of the 60 `eval_v1` query
families (target: 180 rows). It is deliberately family-expanded rather than an
independent held-out benchmark. Do not use its plan, candidates, accepted rows,
or review queue as SFT input.

## Validation policy

1. The plan preserves each source family's split, difficulty, and three axes.
2. Generation must create a new question and new gold SQL; direct duplicates of
   `eval_v1` or other `eval_v2` rows are rejected.
3. Every gold query must execute in BigQuery and return a non-empty result.
4. Gemini Flash Low and Grok 4.6 (its configured low-reasoning setting) are
   evaluated under the identical P1 prompt and execution scorer.
5. A row is automatically accepted only when both frontier outputs are result-
   equivalent to its gold SQL. All other rows enter an explicit manual queue.

The frontier-consensus rule is a pragmatic silver-quality gate, not proof that
gold SQL is correct. Report `eval_v1` as the primary benchmark; label all
`eval_v2` analyses as exploratory/silver.

## 1. Plan (already created)

```powershell
$env:UV_CACHE_DIR = '.codex-tmp\uv-cache'
uv run --offline python src\build_eval_v2_plan.py `
  --out artifacts\eval_v2\plan_v1.jsonl
```

The existing `artifacts/eval_v2/plan_v1.jsonl` contains 180 slots and embeds
private family references solely for generation. Keep it out of training data.

## 2. Generate candidates

Run from an authorized shell; it calls the configured Vertex/partner model.

```powershell
$env:UV_CACHE_DIR = '.codex-tmp\uv-cache'
uv run --offline python src\generate_eval_v2_candidates.py `
  --plan artifacts\eval_v2\plan_v1.jsonl `
  --out artifacts\eval_v2\candidates_grok_v1.jsonl `
  --known eval_v1.jsonl `
  --model grok-4.6 `
  --max-attempts-per-slot 3
```

Rows which exhaust their retries are retained in the adjacent
`candidates_grok_v1.failures.jsonl`. Re-run the same command with `--resume`
to retry only missing slots while preserving already accepted candidates.

## 3. Static and gold-execution gate

```powershell
$env:UV_CACHE_DIR = '.codex-tmp\uv-cache'
uv run --offline python src\validate_eval_v2.py `
  --data artifacts\eval_v2\candidates_grok_v1.jsonl `
  --plan artifacts\eval_v2\plan_v1.jsonl `
  --execute `
  --report artifacts\eval_v2\gold_execution_v1.jsonl
```

Do not advance until this reports all 180 rows as executable and non-empty.

If the report contains execution errors or zero-row gold results, replace only
those slots. The prior candidates remain excluded as direct duplicates:

```powershell
$env:UV_CACHE_DIR = '.codex-tmp\uv-cache'
uv run --offline python src\generate_eval_v2_candidates.py `
  --plan artifacts\eval_v2\plan_v1.jsonl `
  --out artifacts\eval_v2\candidates_grok_v1.jsonl `
  --known eval_v1.jsonl `
  --model gemini-3.8-flash-low `
  --max-attempts-per-slot 5 `
  --resume `
  --replace-from-report artifacts\eval_v2\gold_execution_v1.jsonl
```

Then rerun the gold-execution gate with a new report filename to preserve the
failed attempt as an audit trail. For a systematic empty population, add
`--replacement-instruction` to require a known populated alternative.

## 4. P1 frontier consensus

```powershell
$env:UV_CACHE_DIR = '.codex-tmp\uv-cache'

uv run --offline python src\run_eval.py `
  --eval artifacts\eval_v2\candidates_grok_v1.jsonl `
  --eval-version eval_v2_silver_v1 `
  --condition P1 `
  --model gemini-3.8-flash-low `
  --out eval_results\frontier\gemini-3.8-flash-low\eval_v2_silver_v1\scored\p1.jsonl

uv run --offline python src\run_eval.py `
  --eval artifacts\eval_v2\candidates_grok_v1.jsonl `
  --eval-version eval_v2_silver_v1 `
  --condition P1 `
  --model grok-4.6 `
  --out eval_results\frontier\grok-4.6-low-reasoning\eval_v2_silver_v1\scored\p1.jsonl

uv run --offline python src\triage_eval_v2.py `
  --candidates artifacts\eval_v2\candidates_grok_v1.jsonl `
  --gemini eval_results\frontier\gemini-3.8-flash-low\eval_v2_silver_v1\scored\p1.jsonl `
  --grok eval_results\frontier\grok-4.6-low-reasoning\eval_v2_silver_v1\scored\p1_retry1.jsonl `
  --accepted-out artifacts\eval_v2\accepted_frontier_consensus_v1.jsonl `
  --review-out artifacts\eval_v2\manual_review_v1.jsonl
```

Review every row in `manual_review_v1.jsonl` individually before placing it in
any final expanded evaluation set. Preserve the raw candidates, model results,
accepted rows, and review queue as immutable evidence.

## Finalized silver set

The completed `eval_v2_silver_v1` contains 178 rows:

- 107 rows accepted by Gemini-and-Grok result consensus;
- 71 rows accepted through recorded manual semantic review;
- 2 rows rejected because they asked for leaf categories while their gold SQL
  did not restrict either the population or the benchmark average to leaves.

Use `artifacts/eval_v2/eval_v2_silver_v1.jsonl` for downstream supplementary
reporting. The final gold-execution report is `gold_execution_v3.jsonl`;
all 178 retained rows execute and return a non-empty result. The Grok run used
for the actual triage is `p1_retry1.jsonl` after the initial endpoint timeout.