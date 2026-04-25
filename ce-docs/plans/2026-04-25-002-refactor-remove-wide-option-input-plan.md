---
title: refactor: Remove wide option-chain input support
type: refactor
status: active
date: 2026-04-25
origin: ce-docs/brainstorms/2026-04-25-oipd-raw-data-intake-schema-requirements.md
---

# refactor: Remove wide option-chain input support

## Overview

Remove `call_price` / `put_price` as an accepted option-chain input shape. OIPD should accept one raw option-chain shape: long-form rows with `strike`, `expiry`, `option_type`, and `last_price`, plus optional quote metadata.

This is intentionally narrow. It changes only input-format acceptance and related tests/docs. It must not change parity math, fitting intent, yfinance ingestion, price-selection policy, probability outputs, volatility outputs, or model-output column names.

For a non-SWE mental model: this is removing an extra spreadsheet template from the front door. It does not change the model; it only says users and helper APIs must submit rows in the same long-form layout.

## Problem Frame

The public input contract has already been narrowed in docs/docstrings, but `oipd/core/data_processing/parity.py` still accepts wide rows with one `call_price` and one `put_price` column per strike. That keeps the confusing format alive in executable behavior and tests.

The user decision is now to remove wide support fully, with no deprecation-warning compatibility path. The implementation should remove the wide branch where it is an input format, while preserving legitimate `call_price` / `put_price` names where they are outputs from pricing, reconstruction, or examples.

## Requirements Trace

- R1. Public raw CSV/DataFrame intake has one standard format: long-form option-chain rows.
- R2. Fit-ready user-supplied data requires `strike`, `expiry`, `option_type`, and `last_price` for now.
- R3. `option_type` identifies calls and puts and is normalized to `C` / `P`.
- R4. `call_price` and `put_price` are not accepted or documented as raw input columns.
- R5. No deprecation-warning path: wide input should fail with the same direct unsupported-format style used by existing validation.
- R6. Internal parity logic may still create same-strike call/put pairs as local variables, but it must not accept pre-pivoted wide input.
- R7. Do not change downstream pipeline intent, parity math, fitting behavior, yfinance behavior, or model-output/export columns.
- R8. Update existing tests only; do not add new test files.

## Scope Boundaries

- Do not touch `oipd/core/data_processing/selection.py`.
- Do not change `oipd/pipelines/vol_curve/vol_curve_pipeline.py` or `_legacy/estimator.py` except if a docstring reference is directly wrong.
- Do not change yfinance output shape; it already returns long-form calls/puts with `option_type`.
- Do not rename model-output `call_price` columns in `oipd/pipelines/_internal/reconstruct.py`, probability-density conversion code, pricing code, or notebooks where they represent calculated call prices.
- Do not edit notebook output cells solely to remove displayed `call_price` / `put_price` model outputs.
- Do not add bid/ask-only support or change the current `last_price` requirement.
- Do not add open-interest handling.

## Context & Research

### Relevant Code and Patterns

- `oipd/core/data_processing/parity.py` is the live input-acceptance surface for wide `call_price` / `put_price` data. It branches on those columns in `infer_forward_from_atm`, `apply_put_call_parity_to_quotes`, `detect_parity_opportunity`, and `preprocess_with_parity`.
- `tests/core/data_processing/test_parity.py` includes explicit wide-format coverage: `test_detect_separate_price_format` and `test_infer_forward_separate_price_format`.
- `oipd/core/__init__.py` and `oipd/core/data_processing/__init__.py` re-export parity helpers, so wide input removal is a real public behavior change for those helper APIs, even though it is scoped to input formatting.
- `README.md`, `docs/3_user-guide.md`, `oipd/data_access/sources.py`, `oipd/interface/volatility.py`, and `oipd/interface/probability.py` already describe long-form public input after the prior cleanup.
- `oipd/data_access/vendors/yfinance/reader.py` takes yfinance's separate calls/puts tables and combines them into long-form rows with `option_type`.
- `examples/quickstart_VolCurve.ipynb`, `examples/quickstart_VolSurface.ipynb`, `oipd/pipelines/_internal/reconstruct.py`, and probability/pricing internals use `call_price` / `put_price` as calculated outputs. Those are not wide raw-input support and should remain unchanged.

### Institutional Learnings

- No `ce-docs/solutions/` learnings are present in this worktree.

### External References

- External research was not needed. This is a local API/input-contract cleanup grounded in repo behavior and vendor-data inspection.

## Key Technical Decisions

- Remove wide input at the parity helper boundary rather than adding warnings. This matches the user's decision to have one standard input format and no deprecation errors/warnings.
- Keep the public parity helper names exported. The change is to accepted input shape, not to the public symbol list.
- Preserve local variable names like `call_price` and `put_price` inside parity calculations when they mean the observed call/put premiums extracted from long-form rows. Removing those names would make the math less clear without changing input support.
- Keep calculated output columns named `call_price` where they are model outputs, because those are not raw intake schemas.
- Treat hybrid DataFrames with valid long-form columns plus stray `call_price` / `put_price` as long-form input. The wide columns should be ignored rather than rejected, because rejecting passthrough extra columns would broaden this into general reader validation. The important invariant is that wide columns must never take precedence or act as a fallback price source.
- Assume `option_type` has already been normalized to `C` / `P` by intake paths or caller preparation. This plan should not add new option-type normalization behavior inside parity helpers.

## Open Questions

### Resolved During Planning

- Should wide `call_price` / `put_price` support be kept as internal-only? Resolved: no, remove it fully as an accepted input format.
- Should the pipeline behavior change beyond input-format rejection? Resolved: no.
- Should there be a deprecation-warning period? Resolved: no.
- Should new tests be added? Resolved: no new test files; update existing focused tests only.

### Deferred to Implementation

- Exact error text for unsupported wide input: keep it concise and consistent with existing parity errors, but avoid promising a migration helper.
- Whether any existing parity tests should be deleted versus rewritten as negative assertions. This can be decided while editing `tests/core/data_processing/test_parity.py`.

## Implementation Units

- [ ] **Unit 1: Remove wide input branches from parity helpers**

**Goal:** Make parity helpers accept only long-form option rows, while keeping existing long-form behavior intact.

**Requirements:** R1, R3, R4, R5, R6, R7

**Dependencies:** None

**Files:**
- Modify: `oipd/core/data_processing/parity.py`
- Test: `tests/core/data_processing/test_parity.py`

**Approach:**
- In `infer_forward_from_atm`, remove the top-level branch that iterates rows with `call_price` / `put_price` columns.
- In `apply_put_call_parity_to_quotes`, remove the wide branch that treats one row as both call and put data.
- In `detect_parity_opportunity`, remove the wide branch that counts rows with both wide prices.
- In `preprocess_with_parity`, remove both fallback paths that create `last_price` from `call_price`: the no-opportunity fallback and the post-exception fallback.
- Update docstrings and error messages to describe the only accepted parity input shape: long-form rows with `last_price` and `option_type`.
- Do not reject extra `call_price` / `put_price` columns when valid long-form columns are present; simply ignore them.
- Keep the internal candidate-pair dictionaries and local variables named `call_price` and `put_price` when they are derived from long-form rows.

**Patterns to follow:**
- Existing long-form branch in `oipd/core/data_processing/parity.py`.
- Existing concise `ValueError` style in parity helpers.

**Test scenarios:**
- Happy path: long-form same-strike call/put rows still infer forward price.
- Happy path: long-form same-strike call/put rows still produce one parity-adjusted synthetic call row per usable strike.
- Error path: a DataFrame with `strike`, `call_price`, and `put_price` no longer counts as a parity opportunity.
- Error path: a DataFrame with `strike`, `call_price`, and `put_price` fails parity preprocessing as unsupported/no usable long-form price data.
- Hybrid path: a DataFrame with valid long-form columns plus stray `call_price` / `put_price` uses only `last_price` and `option_type`; wide columns do not override or backfill prices.

**Verification:**
- Existing long-form parity tests continue to pass.
- A repo search confirms `{"call_price", "put_price"}.issubset(options_df.columns)` no longer appears in `oipd/core/data_processing/parity.py`.

- [ ] **Unit 2: Rewrite wide-format parity tests as removal coverage**

**Goal:** Make existing tests prove wide input is gone, without adding new test files.

**Requirements:** R4, R5, R8

**Dependencies:** Unit 1

**Files:**
- Modify: `tests/core/data_processing/test_parity.py`

**Approach:**
- Replace `test_detect_separate_price_format` with a negative test that wide rows are not considered a parity opportunity.
- Replace `test_infer_forward_separate_price_format` with a negative test that wide rows raise the intended unsupported-format/no-pair error.
- Add a negative assertion for `apply_put_call_parity_to_quotes` using a wide-only DataFrame, because that exported helper currently has its own wide-input branch.
- Keep all long-form parity tests and realistic market tests unchanged unless they need minor wording updates.
- Do not broaden tests into bid/ask-only, expiry grouping, or pricing-pipeline behavior; those are outside this input-format cleanup.

**Patterns to follow:**
- Existing pytest style in `tests/core/data_processing/test_parity.py`.

**Test scenarios:**
- Error path: `detect_parity_opportunity` returns `False` for rows containing only `strike`, `call_price`, and `put_price`.
- Error path: `infer_forward_from_atm` rejects rows containing only `strike`, `call_price`, and `put_price`.
- Error path: `apply_put_call_parity_to_quotes` rejects rows containing only `strike`, `call_price`, and `put_price`.
- Error path: `preprocess_with_parity` rejects rows containing only `strike`, `call_price`, and `put_price`.
- Hybrid path: when long-form rows also contain stray `call_price` / `put_price`, parity helpers continue to use `last_price` / `option_type` and do not use the wide columns.
- Regression guard: long-form tests in the same file remain unchanged and passing.

**Verification:**
- The focused parity test file clearly communicates that the supported input shape is long-form only.
- No new test files are created.

- [ ] **Unit 3: Clean parity-facing documentation and comments**

**Goal:** Remove remaining parity-doc wording that presents wide input as supported, without touching broader model-output documentation.

**Requirements:** R1, R4, R6, R7

**Dependencies:** Unit 1

**Files:**
- Modify: `oipd/core/data_processing/parity.py`
- Review only: `README.md`
- Review only: `docs/3_user-guide.md`
- Review only: `oipd/data_access/sources.py`
- Review only: `oipd/interface/volatility.py`
- Review only: `oipd/interface/probability.py`

**Approach:**
- Update parity docstrings that currently say "either call_price/put_price columns or option_type rows."
- Include the `apply_put_call_parity` wrapper docstring, which currently mentions explicit call/put columns.
- Keep public docs/docstrings as-is if they already describe long-form input only.
- Do not alter pricing-output docs or examples where `call_price` means a calculated call option value rather than a raw input column.

**Patterns to follow:**
- Current user guide wording for long-form input in `docs/3_user-guide.md`.
- Current public API docstring wording in `oipd/data_access/sources.py`.

**Test scenarios:**
- Test expectation: none -- documentation/comment cleanup only.

**Verification:**
- Search results for `call_price` / `put_price` are reviewed and classified.
- Any remaining hits are either model-output terminology, local variable names, or historical plan/requirements context, not live accepted input-format documentation.

- [ ] **Unit 4: Add a brief changelog note for the breaking input-format cleanup**

**Goal:** Make the breaking helper-input behavior explicit without creating a deprecation path.

**Requirements:** R4, R5, R7

**Dependencies:** Units 1-3

**Files:**
- Modify: `CHANGELOG.md`

**Approach:**
- Add a short unreleased or current-entry note that `call_price` / `put_price` wide option-chain input is no longer accepted by parity preprocessing helpers.
- State the replacement shape plainly: long-form rows with `option_type` and `last_price`.
- Do not describe a warning period or backward-compatibility shim.

**Patterns to follow:**
- Existing terse bullet style in `CHANGELOG.md`.

**Test scenarios:**
- Test expectation: none -- changelog-only change.

**Verification:**
- Changelog note is present and scoped to input format only.

## System-Wide Impact

- **Interaction graph:** Direct callers of parity helpers through `oipd.core` or `oipd.core.data_processing` that pass wide rows will fail. Long-form callers and pipeline calls should continue unchanged.
- **Error propagation:** Wide rows should now follow the same unsupported/no-usable-price error path as any non-long-form input. No new warning path is planned.
- **State lifecycle risks:** None; this does not mutate stored data or perform migrations.
- **API surface parity:** Public docs, parity helper behavior, and parity tests should all agree on long-form-only input.
- **Integration coverage:** Existing parity tests are sufficient for this isolated helper-input change. Pipeline-level tests should not need changes if the pipeline already passes long-form rows.
- **Unchanged invariants:** yfinance still combines calls and puts into long-form rows; fitted outputs may still expose calculated call prices; pricing math remains unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Removing wide support breaks direct users of exported parity helpers. | Add a clear changelog note and keep the replacement long-form schema explicit. |
| Overzealous search-and-replace renames legitimate calculated `call_price` outputs. | Classify every `call_price` / `put_price` hit; edit only input-format branches/docs/tests. |
| Implementation accidentally changes parity math while removing branches. | Keep long-form branch logic structurally unchanged and rely on existing long-form parity tests. |
| Plan drifts into bid/ask-only or broader pipeline changes. | Scope units only to wide input rejection, docs/comments, existing tests, and changelog. |

## Documentation / Operational Notes

- This is a breaking cleanup for a confusing input shape, not a deprecation sequence.
- The changelog should be the only release-facing note needed.
- No migration utility is planned; users should reshape wide data outside OIPD into long-form rows.

## Sources & References

- **Origin document:** [ce-docs/brainstorms/2026-04-25-oipd-raw-data-intake-schema-requirements.md](../brainstorms/2026-04-25-oipd-raw-data-intake-schema-requirements.md)
- Prior plan: `ce-docs/plans/2026-04-25-001-refactor-raw-data-intake-schema-plan.md`
- Related code: `oipd/core/data_processing/parity.py`
- Related exports: `oipd/core/__init__.py`
- Related exports: `oipd/core/data_processing/__init__.py`
- Related tests: `tests/core/data_processing/test_parity.py`
- Public docs already aligned: `docs/3_user-guide.md`
- Public docs already aligned: `README.md`
- Vendor long-form pattern: `oipd/data_access/vendors/yfinance/reader.py`
