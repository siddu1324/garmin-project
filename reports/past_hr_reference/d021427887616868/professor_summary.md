# Professor briefing — Notebook 10

The earlier-recording HR reference does not improve prediction in this bounded experiment.
About 53.5% of development composite-score variation is within participants and 46.5% between
observed person means. Retrospective person-centered HR associations are small and uncertain;
these means never enter prediction.

The fixed past-only rule retains 1,575/1,634 existing 60-minute HR responses, from 116 people.
Reference history is [cutoff−25h,cutoff−1h), averaging observed UTC-minute means, requiring 30
occupied minutes across three UTC hours. This is an earlier-recording reference, not resting HR.
Eligible histories have median 572 occupied minutes but an 11.46-hour largest gap including edges.

On identical outer folds, HGB reference/deviation E MAE is 2.0949, versus raw HR C 2.0646 and
reference-only D 2.0854. E−C: +.0303 [−.0253,+.0840]; E−D: +.0095 [−.0172,+.0343].
Ridge E is also worse in point estimate. Matching median MAE is 2.0673. Conditional bootstrap
intervals and nested tuning do not remove repeated exploratory development limitations.

Do not expand models automatically. Request acquisition/receipt/batching definitions, prompt logs,
sensor-quality flags and IBI anchoring/missed-beat semantics before another measurement-valid
experiment. The request is drafted locally only. All 32 final-test participants remain sealed.
