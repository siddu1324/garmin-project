# Matched context-control amendment

Written after inspecting the completed 600 primary-stage fits, before computing these additional
controls. The original specification and results remain intact. This is a disclosed comparison
correction, not an independently prospectively specified new discovery experiment.

The contextual compact weighted/over/under/conditional models require context-only counterparts
with the same training procedure to isolate the contribution of physiological coordinates. Comparing
them only with unweighted context conflates rebalancing with feature information, even on identical
EMA cohorts. This affects all four rebalanced contextual arms, not just the highest observed model.

Add exactly four context-only specifications (balanced weights, random over, random under, and
conditional donor-context copying), for both targets and all five unchanged outer folds: 40 fits.
Reuse the original 12-candidate logistic grid, grouped inner folds, seed, transforms and threshold rule.
No new feature, sampler, quality rule, model family or threshold objective is introduced.

For conditional augmentation, context comes unchanged from the base donor, exactly as in the
paired contextual physiological model. With zero physiological coordinates there is nothing to
interpolate; these rows are duplicates, not synthetic physiology. Base donors are chosen before
neighbor sampling, so the context rows/labels match the physiological arm. Verify original and
resampled counts, outer/inner membership, and base-donor row multisets for all four comparisons.

Report each contextual compact model versus its matching context control. Preserve the unweighted
context comparisons and label the new contrasts. Do not revise primary conclusions, select an outer
winner, or call the added controls independent confirmation. Save source/configuration hashes in a
separate immutable control-stage bundle and preserve the initial aggregate results privately.
