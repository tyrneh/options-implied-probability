---
date: 2026-04-25
topic: oipd-raw-data-intake-schema
---

# OIPD Raw Data Intake Schema

## Problem Frame

OIPD's public CSV/DataFrame intake contract should match how users normally receive option-chain data: long-form rows, one row per option contract. The current surface is broader than necessary because `call_price` / `put_price` are accepted in parity preprocessing and covered by tests, making an internal paired call/put representation look like a supported vendor input shape.

The desired outcome is a narrower, easier-to-explain public contract: users provide long-form option-chain data; OIPD handles pairing, parity preprocessing, and synthetic call conversion internally when needed.

## Requirements

**Public Intake Contract**

- R1. Public docs and user-facing docstrings must describe long-form CSV/DataFrame input as the single standard raw-data format.
- R2. Fit-ready user-supplied data must require `strike`, `expiry`, and `option_type`, plus a supported price source.
- R3. Supported public price sources must be either `bid` + `ask` or `last_price`; `bid` + `ask` is preferred, while `last_price` is acceptable with `volume` recommended.
- R4. `option_type` must identify calls and puts and be normalized to `C` / `P`.

**Column Scope**

- R5. `call_price` and `put_price` must not be presented as public raw-data intake columns.
- R6. `open_interest` must not be part of the current standard input format.
- R7. `volume` must remain part of the supported public schema because downstream fitting can use it for weighting.
- R8. `last_trade_date` must remain part of the supported public schema because downstream filtering can use it to remove stale quotes.

**Internal Processing Boundary**

- R9. Internal parity preprocessing may create or consume paired call/put intermediate data as needed, but that intermediate representation must not be documented as user-facing intake.
- R10. Tests should make the public expectation clear: raw user intake is long-form and relies on `strike`, `expiry`, `option_type`, and a supported price source.

## Success Criteria

- Public docs and docstrings describe one standard long-form input for user-supplied CSV/DataFrame intake.
- Fit-ready user data does not require or mention `call_price`, `put_price`, or `open_interest`.
- Tests distinguish public intake behavior from internal parity preprocessing behavior.
- Existing long-form vendor and broker-style data remains easy to load with column mapping.

## Scope Boundaries

- This change does not require removing all internal uses of `call_price` or `put_price`; pricing outputs and fitted result exports may still use those terms where they represent model outputs rather than raw input columns.
- This change does not decide future `open_interest` support; it can be added later as an explicit input column if there is a user-facing need.
- This change does not define a new vendor abstraction or a new data storage format.
- This change does not introduce a deprecation path or warning-only compatibility mode for alternative public input formats.

## Key Decisions

- Public intake should have one standard long-form format: this matches common vendor and broker data and reduces the amount users need to understand before fitting.
- Wide-form `call_price` / `put_price` should be excluded from the public contract: this shape is closer to an internal parity/pre-pivoted representation than a natural user input format.
- `open_interest` should be ignored for now: if it becomes useful later, it can be added deliberately as a standard input column.
- Unsupported public input shapes should not receive a deprecation-error pathway; the contract should simply point users to the standard long-form format.

## Dependencies / Assumptions

- Current code inspection found user-facing standard-name docs in `oipd/data_access/sources.py`, reader validation and normalization in `oipd/data_access/readers/base.py`, yfinance metadata mapping in `oipd/data_access/vendors/yfinance/reader.py`, and parity wide-form support in `oipd/core/data_processing/parity.py`.
- Current tests include explicit wide-form parity coverage in `tests/core/data_processing/test_parity.py`; planning should decide whether those tests become internal-only coverage or are removed.

## Outstanding Questions

### Resolve Before Planning

- None.

### Deferred to Planning

- [Affects R5, R9][Technical] Decide whether parity functions should keep accepting `call_price` / `put_price` as private/internal inputs or fully remove that branch.

## Next Steps

-> `$ce:plan` for structured implementation planning.
