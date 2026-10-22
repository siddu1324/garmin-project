"""Timestamp-aware descriptive features; no resampling, beat inference or sensor filling."""
import numpy as np
from scipy.stats import kurtosis, skew

STAT = ['mean', 'median', 'minimum', 'maximum', 'sd', 'variance', 'p10', 'p25', 'p75',
        'p90', 'range', 'iqr', 'mad', 'cv', 'first', 'latest', 'slope', 'net_change',
        'half_mean_change', 'skewness', 'excess_kurtosis', 'mean_absolute_change',
        'median_absolute_change', 'rms_change', 'observed_minute_mean', 'observed_minute_sd',
        'rolling5_mean', 'rolling5_min', 'rolling5_max', 'rolling5_within_sd']
ACTIVITY = ['added_supported', 'supported_rate', 'median_pair_rate', 'p90_pair_rate',
            'max_pair_rate', 'sd_pair_rate', 'zero_change_supported_fraction']
QUALITY = ['recorded_groups', 'accepted_readings', 'accepted_fraction', 'occupied_minutes',
           'occupied_fraction', 'span_seconds', 'latest_age_seconds', 'left_gap_seconds',
           'maximum_internal_gap_seconds', 'maximum_gap_seconds', 'valid_adjacent_pairs']


def slice_window(g, start, end):
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError('Invalid observation window.')
    a, b = np.searchsorted(g.n_epoch_s.to_numpy(float), [start, end], side='left')
    return g.iloc[a:b], int(a), int(b)


def moments(v):
    q10, q25, q75, q90 = np.quantile(v, [.1, .25, .75, .9])
    mu, sd = v.mean(), v.std(ddof=1)
    return dict(mean=mu, median=np.median(v), minimum=v.min(), maximum=v.max(), sd=sd,
                variance=sd**2, p10=q10, p25=q25, p75=q75, p90=q90,
                range=np.ptp(v), iqr=q75-q25, mad=np.median(np.abs(v-np.median(v))),
                cv=sd/mu if mu != 0 else np.nan)


def supported(t, n=2, span=1):
    return len(t) >= n and t[-1]-t[0] >= span


def observed_features(g, start, end, cfg):
    w, a, b = slice_window(g, start, end)
    good = w.a_acceptable.to_numpy(bool)
    all_t, all_v = w.n_epoch_s.to_numpy(float), w.n_value.to_numpy(float)
    t, v = all_t[good], all_v[good]
    f = dict.fromkeys(STAT, np.nan)
    reason = dict.fromkeys(STAT, 'no_recording' if not len(w) else
                           'all_rejected' if not len(t) else 'insufficient_support')
    if len(t):
        f.update(first=v[0], latest=v[-1])
    if supported(t, cfg['summary_min_count'], cfg['summary_min_span_seconds']):
        f.update(moments(v), net_change=v[-1]-v[0])
    if supported(t, cfg['slope_min_count'], cfg['slope_min_span_seconds']):
        x = (t-t[0])/60
        x -= x.mean()
        f['slope'] = np.dot(x, v-v.mean()) / np.dot(x, x)
    if supported(t, cfg['shape_min_count'], cfg['shape_min_span_seconds']) and np.ptp(v) > 0:
        f.update(skewness=skew(v, bias=False), excess_kurtosis=kurtosis(v, bias=False))
    half = (start+end)/2
    if supported(t[t < half]) and supported(t[t >= half]):
        f['half_mean_change'] = v[t >= half].mean()-v[t < half].mean()
    pair = good[1:] & good[:-1] & (np.diff(all_t) > 0) & (
        np.diff(all_t) <= cfg['adjacent_max_gap_seconds'])
    d = np.diff(all_v)[pair]
    if len(d) >= cfg['adjacent_min_pairs']:
        f.update(mean_absolute_change=np.abs(d).mean(), median_absolute_change=np.median(np.abs(d)),
                 rms_change=np.sqrt(np.mean(d*d)))
    if len(t):
        utc_bin = np.floor(t/60).astype(np.int64)
        _, inv = np.unique(utc_bin, return_inverse=True)
        means = np.bincount(inv, weights=v)/np.bincount(inv)
        if len(means) >= 2:
            f.update(observed_minute_mean=means.mean(), observed_minute_sd=means.std(ddof=1))
        bins = np.floor((t-start)/60).astype(np.int64)
        count = int(round((end-start)/60))
        sums = np.bincount(bins, weights=v, minlength=count)
        nums = np.bincount(bins, minlength=count)
        minute = np.divide(sums, nums, out=np.full(count, np.nan), where=nums > 0)
        means5, sds5 = [], []
        for j in range(max(0, count-cfg['rolling_minutes']+1)):
            m = minute[j:j+cfg['rolling_minutes']]
            m = m[np.isfinite(m)]
            if len(m) >= cfg['rolling_min_occupied_minutes']:
                means5.append(m.mean())
                sds5.append(m.std(ddof=1))
        if means5:
            f.update(rolling5_mean=np.mean(means5), rolling5_min=min(means5),
                     rolling5_max=max(means5), rolling5_within_sd=np.mean(sds5))
    for k, value in f.items():
        if np.isfinite(value):
            reason[k] = 'available'
        elif k in ['skewness', 'excess_kurtosis'] and len(v) and np.ptp(v) == 0:
            reason[k] = 'constant_values'
    occupied = len(np.unique(np.floor((t-start)/60)))
    q = dict(recorded_groups=len(w), accepted_readings=len(t),
             accepted_fraction=len(t)/len(w) if len(w) else np.nan,
             occupied_minutes=occupied, occupied_fraction=occupied/((end-start)/60),
             span_seconds=t[-1]-t[0] if len(t) else np.nan,
             latest_age_seconds=end-t[-1] if len(t) else np.nan,
             left_gap_seconds=t[0]-start if len(t) else np.nan,
             maximum_internal_gap_seconds=np.diff(t).max() if len(t) > 1 else np.nan,
             maximum_gap_seconds=np.diff(np.r_[start, t, end]).max(), valid_adjacent_pairs=len(d))
    return f, q, reason, (a, b)


def activity_features(g, start, end):
    w, a, b = slice_window(g, start, end)
    p = w[w.a_pair_accepted & w.a_pair_start.ge(start)]
    f = dict.fromkeys(ACTIVITY, np.nan)
    q = dict(supported_pairs=len(p), supported_seconds=p.a_pair_duration.sum(),
             reset_pairs=int(((w.a_pair_rejection_mask.to_numpy(int) & 128) != 0).sum()),
             boundary_pairs=int(w.a_pair_start.lt(start).sum()), recorded_groups=len(w))
    if len(p):
        delta, dt = p.a_pair_delta.to_numpy(float), p.a_pair_duration.to_numpy(float)
        rate = 60*delta/dt
        f.update(added_supported=delta.sum(), supported_rate=60*delta.sum()/dt.sum(),
                 median_pair_rate=np.median(rate), p90_pair_rate=np.quantile(rate, .9),
                 max_pair_rate=rate.max(), zero_change_supported_fraction=dt[delta == 0].sum()/dt.sum())
        if len(p) >= 2:
            f['sd_pair_rate'] = rate.std(ddof=1)
    return f, q, (a, b)


def coobserved(hr, steps, start, end):
    """Overlap-conditioned observations, not a causal activity adjustment."""
    w, _, _ = slice_window(steps, start, end)
    pairs = w[w.a_pair_accepted & w.a_pair_start.ge(start)]
    h, _, _ = slice_window(hr, start, end)
    h = h[h.a_acceptable]
    ts, vs = h.n_epoch_s.to_numpy(float), h.n_value.to_numpy(float)
    values = {}
    for positive, name in [(False, 'hr_zero_step_mean'), (True, 'hr_positive_step_mean')]:
        mask = np.zeros(len(ts), bool)
        for p in pairs[pairs.a_pair_delta.gt(0).eq(positive)].itertuples():
            mask |= (ts >= p.a_pair_start) & (ts < p.n_epoch_s)
        values[name] = vs[mask].mean() if supported(ts[mask], 3, 60) else np.nan
    values['hr_step_stratum_difference'] = values['hr_positive_step_mean']-values['hr_zero_step_mean']
    return values
