# Provider questions — draft only, not sent

1. Does Garmin `ts` mark acquisition, device buffering, phone receipt or server upload? Can batching,
   timestamp rounding or retransmission alter spacing/order? How were minute-resolution EMA starts recorded?
2. Does each IBI timestamp mark the interval start or end? Are units milliseconds? How are missed,
   extra, duplicate or rejected beats represented, and are disconnected recordings explicitly marked?
3. Which devices and sensor modalities produced HR/IBI? Are wear, motion, beat-quality, ectopic-beat
   or normal-beat annotations available? Does HR represent an instantaneous reading or a prior average?
4. Are steps and floorsClimbed cumulative daily counters? What resets them, in which timezone, and
   can counters decrease/correct or reset invisibly between exports? Are updates periodic or change-triggered?
5. What precisely do calories and intensityMinutes accumulate, in which units and reset period?
   Do calories include resting energy, and are intensity minutes weighted? Can each update be given
   an explicit measurement interval so boundary-crossing increments can be attributed safely?

No participant records, IDs, dates or example source rows are included. This is a local draft only.
