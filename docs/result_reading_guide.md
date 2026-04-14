# Result Reading Guide

## Read In Order

For any official filled matrix run, read outputs in this order:

1. `paper_table.md`
2. `paper_table_category_mean.md`
3. `paper_table_by_axis.md`
4. `failure_analysis/summary.json`
5. `matrix_results.json`

## `paper_table.md`

Read this first to answer:

- Did all official baseline combinations run?
- What are the per-dataset results for each method pairing?
- Are any rows obviously broken, collapsed, or runtime-dominated?

This file is the fastest per-dataset scoreboard. It is useful for checking coverage and spotting outliers, but it is not yet the best file for deciding the research mainline across categories.

## `paper_table_category_mean.md`

Read this next for first-pass research judgment.

It collapses per-dataset records into one category-mean row per method pairing, which makes it the clearest answer to:

- Which baselines are strongest on average across `bottle`, `capsule`, and `grid`?
- Are diffusion baselines competitive enough to stay on the mainline?
- Is a gain broad or just driven by one category?

If you only have time to read one result file for actual research direction, start here.

## `paper_table_by_axis.md`

Read this when you need structured comparison by research question rather than by raw row.

It groups methods by axis buckets such as:

- classical vs diffusion static
- diffusion static vs diffusion process
- unconditional vs conditional

Use it to judge whether a family-level idea is paying off. This is especially important when deciding whether `diffusion_basic` or `diffusion_inversion_basic` deserves more work.

## `failure_analysis/summary.json`

Read this after the tables, not before them.

Start with:

- which method groups exported cases
- which run each group came from
- how many ranked cases were exported per group

Then drill into the case folders only when a table result needs explanation. Failure analysis is an audit and interpretation layer. It should explain a result, not replace the contract-level comparison.

## Which Outputs Support Research Judgment

These outputs can support research decisions:

- `paper_table_category_mean.md`
- `paper_table_by_axis.md`
- `paper_table.md` as a per-dataset check
- `matrix_results.json` when you need raw run-level confirmation

These outputs should be treated as smoke or audit support only:

- Smoke matrix results from `paper_baseline_matrix_official_v1_filled_smoke`
- `failure_analysis/summary.json` and the exported case bundles
- Individual run folders under `outputs/runs/`

Smoke confirms the pipeline and contracts. It does not justify a research conclusion. Failure analysis explains behavior. It does not establish the ranking by itself.
