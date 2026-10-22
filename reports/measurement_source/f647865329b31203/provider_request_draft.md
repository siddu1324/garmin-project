# Provider request draft — not sent

We can trace our analysis back to the supplied CSVs, but need to establish what the exported clocks
and IBI rows mean. Could you provide this small provenance packet for **the build that produced these
exports**, distinguishing it from the proposed STARRS/Vivoactive 5 integration?

1. **Implementation identity:** collection-app name/platform/build or Git commit; Garmin product
   (SDK edition or API), exact version, class/callback or endpoint; dependency-lock/build excerpt;
   device model, firmware and legacy/Enhanced BBI mode. Please include the relevant collaboration
   letter or collection protocol if it identifies this implementation.
2. **Clock and serialization mapping:** the code assigning `ts` and `dte_tme`, receiving callbacks,
   storing events and exporting `data_type/val`. Is each clock acquisition/event, interval start/end,
   callback, phone/server receipt, or upload time? Are device-to-UTC mapping, precision/rounding,
   clock changes and delayed delivery documented? If using arrays, which order and per-item
   offsets/indices are supplied and which clock is copied into each CSV row?
3. **One small de-identified trace:** original payload → callback/storage representation → exported
   rows, including an ordinary sequence, a batch/retry and a gap or rejected measurement. Keep
   interval order, units, event/batch IDs, offsets and device/receipt clocks; pseudonymize identity
   and consistently shift absolute dates. Please omit credentials and personal survey responses.
4. **Measurement semantics:** confirm HR/IBI units, PPG versus another source, smoothing/caching,
   duplicate/stale/sentinel behavior, missed/withheld/rejected beats and any upstream cleaning or
   reconstructed intervals. Which per-beat confidence, motion/contact/wear or rhythm-quality flags
   were available, and were any lost during export? Are consecutive exported intervals necessarily
   consecutive actual beats? Separately, confirm step-counter reset/update and interval semantics.
5. **Recall alignment:** provide the survey version and intended meaning of “since last prompt,”
   plus scheduled, delivered/displayed, expired/suppressed and submission event logs, including
   unanswered prompts. Previous completed EMAs are not a substitute for these logs. Confirm actual
   start precision/rounding and timezone conventions; reported Morning wake time is retrospective.

The goal is to validate measurement provenance before further HRV or model work, not to choose
thresholds that retain more records. A release-specific contract and tiny redacted example are more
useful than another bulk export without lineage.
