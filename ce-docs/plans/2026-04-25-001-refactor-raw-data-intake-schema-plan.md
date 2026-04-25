---
title: refactor: Standardize raw option-chain intake schema
type: refactor
status: completed
date: 2026-04-25
origin: ce-docs/brainstorms/2026-04-25-oipd-raw-data-intake-schema-requirements.md
---

# refactor: Standardize raw option-chain intake schema

## Overview

Standardize only the isolated public raw-data intake contract for user-supplied CSV/DataFrame data. The public format should be long-form option-chain data: one row per option contract, with explicit `strike`, `expiry`, `option_type`, `last_price`, and supported quote metadata.

This plan does not change parity preprocessing, price selection, fitting behavior, yfinance fetching behavior, or downstream pipeline behavior, except for the narrow volume casing bug fix that preserves lowercase `volume` through parity output.

## Problem Frame

OIPD should be easier to explain to users: public CSV/DataFrame intake should have one standard shape, not a mix of user-facing raw data and internal/pre-pivoted formats. Users should provide ordinary long-form option-chain data and OIPD can continue using its existing downstream internals as-is.

For a non-SWE mental model: this is like standardizing the data-entry form. We are not changing the econometric model; we are only making clearer what columns a user should submit.

## Requirements Trace

- R1. Public docs and user-facing docstrings must describe long-form CSV/DataFrame input as the single standard raw-data format.
- R2. Fit-ready user-supplied data must require `strike`, `expiry`, `option_type`, and `last_price`.
- R3. `bid` and `ask` may be documented as preferred/recommended quote metadata when available, but not as a standalone replacement for `last_price` in this pass.
- R4. `option_type` must identify calls and puts and be normalized to `C` / `P`.
- R5. `call_price` and `put_price` must not be presented as public raw-data intake columns.
- R6. `open_interest` must not be part of the current standard input format.
- R7. `volume` remains the public input column name, lowercase, because downstream fitting can use it for weighting.
- R8. `last_trade_date` remains part of the public input schema because downstream filtering can use it to remove stale quotes.
- R9. Internal parity preprocessing is out of scope for this implementation pass.
- R10. No new tests should be added for this pass.

## Scope Boundaries

- Do not change parity preprocessing in `oipd/core/data_processing/parity.py`, except to fix the existing `Volume` -> `volume` casing bug.
- Do not change price selection in `oipd/core/data_processing/selection.py`.
- Do not change fitting pipelines in `oipd/pipelines/`.
- Do not add new tests.
- Do not add a deprecation-warning path for `call_price` / `put_price`.
- Do not add `open_interest` as metadata or a supported input column.
- Do not rename fitted-result or pricing-output columns that use `call_price` / `put_price` as model outputs.
- Do not edit notebook output cells solely because fitted result examples contain model-output `call_price` / `put_price`.

## Context & Research

### Relevant Code and Patterns

- `oipd/data_access/sources.py` exposes `from_csv` and `from_dataframe`; its `from_csv` docstring currently lists `open_interest` among standard names.
- `oipd/data_access/readers/base.py` is the shared reader used by CSV/DataFrame intake and vendor reader paths, so any validation change here must avoid accidentally changing yfinance behavior.
- `oipd/interface/volatility.py` and `oipd/interface/probability.py` include user-facing column-mapping docstrings that should align with the standard raw input format.
- `docs/3_user-guide.md` is the best place to explain the one public raw input schema.
- `README.md` should remain brief and point users to the user guide.
- `oipd/core/data_processing/parity.py` contains wide `call_price` / `put_price` support, but that is now explicitly out of scope for this isolated raw-input cleanup. It also emits uppercase `Volume` in parity output, while fitting reads lowercase `volume`; fixing that casing mismatch is in scope.

### Institutional Learnings

- No relevant `ce-docs/solutions/` learnings were present in this worktree.

### External References

- External research was not needed. This is a local documentation and intake-contract cleanup.

## Key Technical Decisions

- Keep this pass isolated to raw input contract documentation, the narrow volume casing bug fix, and, only if safe, local CSV/DataFrame intake validation.
- Do not touch downstream pipeline behavior beyond preserving lowercase `volume`.
- Do not add tests.
- Use lowercase `volume` as the public input column name.
- Exclude `open_interest` from the public input contract for now.
- Remove `call_price` / `put_price` from public intake docs/docstrings, while leaving internal parity code untouched in this pass.

## Open Questions

### Resolved During Planning

- Should this pass change parity or price-selection behavior? Resolved: no.
- Should new tests be added? Resolved: no.
- Should `open_interest` be preserved or documented now? Resolved: no.

### Deferred to Implementation

- Whether reader validation can be tightened without impacting yfinance or existing tests. If not, keep validation changes out of this pass.
- Exact public wording for bid/ask support, based on current behavior: `last_price` is required for now; `bid` and `ask` are preferred/recommended when available.

## Implementation Units

- [x] **Unit 1: Update public raw-input documentation**

**Goal:** Make the public docs describe one long-form raw input format and remove confusing public mentions of `call_price`, `put_price`, and `open_interest`.

**Requirements:** R1, R2, R3, R5, R6, R7, R8, R10

**Dependencies:** None

**Files:**
- Modify: `docs/3_user-guide.md`
- Modify: `README.md`

**Approach:**
- Add or tighten a concise schema section in `docs/3_user-guide.md`.
- Describe public CSV/DataFrame intake as long-form, one row per option contract.
- List `strike`, `expiry`, `option_type`, and `last_price` as required for now.
- List `bid`, `ask`, `volume`, and `last_trade_date` as supported/recommended columns when available.
- Do not describe `call_price`, `put_price`, or `open_interest` as public input columns.
- Keep `README.md` short and direct users to the user guide for manual CSV/DataFrame upload details.

**Test scenarios:**
- Test expectation: none -- documentation-only change and user explicitly requested no new tests.

**Verification:**
- Manual search confirms public docs no longer present `call_price`, `put_price`, or `open_interest` as raw input schema columns.
- Manual review confirms docs do not promise bid/ask-only fitting behavior.

- [x] **Unit 2: Align user-facing docstrings**

**Goal:** Make public API docstrings match the same single long-form raw input contract without changing runtime behavior.

**Requirements:** R1, R2, R3, R5, R6, R7, R8, R10

**Dependencies:** Unit 1

**Files:**
- Modify: `oipd/data_access/sources.py`
- Modify: `oipd/interface/volatility.py`
- Modify: `oipd/interface/probability.py`

**Approach:**
- Update `sources.from_csv` and `sources.from_dataframe` docstrings to describe the standard input names.
- Remove `open_interest` from standard-name docstrings.
- Update fit/constructor docstrings that mention column mappings so they point users toward the same standard names.
- Avoid code changes unless they are strictly required to keep docstrings accurate.

**Test scenarios:**
- Test expectation: none -- docstring-only change and user explicitly requested no new tests.

**Verification:**
- Manual search confirms public docstrings no longer list `open_interest`, `call_price`, or `put_price` as input schema columns.

- [x] **Unit 3: Optional minimal local-intake validation cleanup**

**Goal:** If it can be done without touching downstream behavior or tests, make local CSV/DataFrame reader errors better reflect the documented standard schema.

**Requirements:** R2, R4, R5, R6, R10

**Dependencies:** Units 1 and 2

**Files:**
- Review/modify only if safe: `oipd/data_access/readers/base.py`

**Approach:**
- Treat this as optional and conservative.
- Do not change validation in a way that breaks yfinance or requires new tests.
- If validation is adjusted, keep it limited to clearer local raw-input errors and explicit option-type normalization.
- Do not make bid/ask-only promises unless current behavior already supports them.
- Do not add or modify pipeline behavior to make new schema combinations work.

**Test scenarios:**
- Test expectation: none -- no new tests by request. Any validation change should be small enough to be covered by existing tests.

**Verification:**
- Existing test suite should not require new test files or new test cases.
- Manual review confirms no changes were made to parity, selection, fitting, or vendor fetch behavior.

- [x] **Unit 4: Fix lowercase volume preservation through parity**

**Goal:** Preserve the public lowercase `volume` column name after parity preprocessing so downstream fitting can see it.

**Requirements:** R7

**Dependencies:** None

**Files:**
- Modify: `oipd/core/data_processing/parity.py`

**Approach:**
- Change parity output from uppercase `Volume` to lowercase `volume`.
- Do not change parity pricing, selection, fitting, or volume weighting logic.

**Test scenarios:**
- Test expectation: none -- user requested no new tests. Existing tests may be run for verification.

**Verification:**
- Manual search confirms parity output writes lowercase `volume`.
- Existing focused parity tests should continue to pass.

## System-Wide Impact

- **Interaction graph:** Public documentation and docstrings become narrower and clearer; downstream data processing remains unchanged except that parity output preserves lowercase `volume`.
- **Error propagation:** No intended changes unless optional local-intake validation cleanup is safe.
- **State lifecycle risks:** None.
- **API surface parity:** Docs and docstrings should agree on the single public raw input format.
- **Integration coverage:** No new tests in this pass by explicit request.
- **Unchanged invariants:** Parity pricing behavior, price selection, fitting behavior, yfinance fetch behavior, model output columns, and probability/volatility object behavior remain unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Docs promise behavior the pipeline does not support. | Require `last_price` for now; present bid/ask as preferred/recommended metadata, not as a standalone replacement. |
| Tightening shared reader validation accidentally breaks yfinance. | Keep reader validation cleanup optional; skip it if it would touch vendor behavior or require tests. |
| Users still discover internal `call_price` / `put_price` support in parity docs. | This pass removes public-intake guidance only; internal parity docs can be handled separately if needed. |
| No new tests means regressions are less guarded. | Keep changes docs/docstrings-first and avoid pipeline behavior changes. |

## Documentation / Operational Notes

- This pass is a public contract cleanup plus one narrow volume casing bug fix, not a pipeline refactor.
- No release note is required unless implementation changes runtime validation behavior.
- No rollout flag, deprecation period, or migration script is planned.

## Sources & References

- **Origin document:** [ce-docs/brainstorms/2026-04-25-oipd-raw-data-intake-schema-requirements.md](../brainstorms/2026-04-25-oipd-raw-data-intake-schema-requirements.md)
- Related docs: `docs/3_user-guide.md`
- Related docs: `README.md`
- Related code: `oipd/data_access/sources.py`
- Related code: `oipd/interface/volatility.py`
- Related code: `oipd/interface/probability.py`
- Optional local-intake code: `oipd/data_access/readers/base.py`
