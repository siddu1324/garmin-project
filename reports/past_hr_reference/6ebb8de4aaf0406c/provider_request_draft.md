# Provider request draft — not sent

Could you clarify the following export semantics and available metadata?

1. For Garmin `ts`, is the timestamp acquisition time, device event time, interval start/end, or server receipt? How are clock drift, timezone changes and delayed uploads represented?
2. Can records arrive in batches or be reordered/repeated? Is a separate original acquisition timestamp available, including precision and latency?
3. Can scheduled, notification/delivery/display and actual survey-start/submission logs identify the preceding prompt for the questionnaire recall period?
4. Which device wear/contact, motion, confidence and HR/beat-quality flags are available, and what do missing/invalid/stale measurements look like in the export?
5. For IBI, does `ts` anchor the start or end of an interval, or receipt? Are intervals beat-to-beat/normal-to-normal, how are missed or rejected beats encoded, and are corrected values or reconstructions applied upstream?

This is a local draft only. No provider message has been sent.
