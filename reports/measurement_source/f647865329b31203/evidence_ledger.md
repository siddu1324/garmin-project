# Exact local evidence anchors

Every source hash is retained in [audit_summary.json](audit_summary.json). Source citations refer to the code/report version pinned by this checkpoint.

| Claim | Local source and line | Established scope |
| --- | --- | --- |
| csv_container | [src/garmin_inventory/sources.py: 62](../../../src/garmin_inventory/sources.py#L62) | Reads ZIP/CSV exports; not collection-app or device parser code. |
| raw_schema | [reports/inventory.json: 61](../../../reports/inventory.json#L61) | Three raw schemas; union has six fields and no explicit SDK/version/beat-quality field. |
| clock_parse | [src/garmin_preprocessing/normalize.py: 158](../../../src/garmin_preprocessing/normalize.py#L158) | Parses Unix seconds without timezone shift; does not verify event-clock meaning. |
| derived_quality | [src/garmin_preprocessing/normalize.py: 168](../../../src/garmin_preprocessing/normalize.py#L168) | q_ flags are analysis-derived; normal-beat status explicitly unverified. |
| raw_duplicates | [src/garmin_preprocessing/normalize.py: 51](../../../src/garmin_preprocessing/normalize.py#L51) | Exact raw/schema duplicates collapse only in logical view; physical copies survive. |
| logical_order | [src/garmin_preprocessing/normalize.py: 261](../../../src/garmin_preprocessing/normalize.py#L261) | Logical ordering uses exported clocks/source row, not a recovered beat sequence. |
| interval_geometry | [src/garmin_preprocessing/eligibility.py: 40](../../../src/garmin_preprocessing/eligibility.py#L40) | Whole-interval containment assumes either start or end anchoring. |
| corrected_windows | [src/garmin_preprocessing/eligibility.py: 31](../../../src/garmin_preprocessing/eligibility.py#L31) | Current corrected generator; historical long-window results remain revoked. |
| stale_versions | [src/garmin_preprocessing/provenance.py: 21](../../../src/garmin_preprocessing/provenance.py#L21) | Rejects known-invalid versions and manifest copies. |
| precision | [reports/hrv_input_validation/30a0bbe188d56b40/report.md: 117](../../../reports/hrv_input_validation/30a0bbe188d56b40/report.md#L117) | Reused stream-level precision evidence; no obsolete 12/24-hour counts reused. |
| package_limits | [reports/ibi_pilot/78af22e7f0ead75b/report.md: 161](../../../reports/ibi_pilot/78af22e7f0ead75b/report.md#L161) | Numerical agreement in limited pilot does not validate physiology. |
| raw_hr_trace | [reports/window_benchmark/e3e34a2ea014f38f/report.md: 81](../../../reports/window_benchmark/e3e34a2ea014f38f/report.md#L81) | Existing independent raw trace and synthetic positive control; no new participant sample. |

Study document: `R61-R33_ResearchStrategy.pdf`, PDF pp. 5 (B2.2) and 7 (B3.4); SHA-256 `cad3443ed2dc340830b20563d36deeb685ec126a4249881e5b34ebbbf66462ff`. The private evidence bundle records the exact local path and page extracts. These describe a study and future plan, not the exported payload contract.
