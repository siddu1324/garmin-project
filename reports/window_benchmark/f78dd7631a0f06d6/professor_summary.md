# Professor briefing — Notebook 09

The new checks do not establish a useful wearable prediction advantage. Wake and 60-minute HR
windows contain identical readings for 116/227 matched Morning responses and share the latest
reading for all 227; wake and 120 minutes match in 173/227. Distinct boundaries often add no data.

A deterministic trace verifies 320 feature values against 18,153 original readings with no mismatch.
A separate synthetic positive control recovers injected signal through the actual linkage/feature/
model pipeline. These checks support implementation correctness but do not validate physiology.

On the fixed 60-minute cohort (1,634 responses, 118 development participants), nested HGB context+HR
MAE is 2.081 versus median 2.071: difference +.010 [−.016,+.039]. It varies predictions but adds no
clear value beyond HGB context (+.007 [−.020,+.036]). Secondary supported steps (1,355 responses,
113 people) add essentially nothing: +.0008 [−.0039,+.0059] versus matched HGB context+HR.
Tuned Ridge also fails to establish improvement. These are loss-and-model-family comparisons,
not an isolated test of nonlinearity; intervals condition on repeated exploratory development fits.

Recommended next step: obtain timestamp/batching, previous-prompt logs and sensor-quality metadata,
then freeze a measurement-valid comparison. Do not expand the grid or open final-test data.
We can test weighting choices ourselves; acquisition semantics, delivery logs and device flags
require provider evidence. Questions are drafted in the report only and have not been sent.
