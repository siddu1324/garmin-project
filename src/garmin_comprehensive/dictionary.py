"""Explicit feature registry and evidence status; counts are never a design objective."""
from .features import STAT, ACTIVITY

FORMULAS = {
 'mean':'sum(x)/n', 'median':'median(x)', 'minimum':'min(x)', 'maximum':'max(x)',
 'sd':'sqrt(sum((x-mean)^2)/(n-1))', 'variance':'sum((x-mean)^2)/(n-1)',
 'p10':'linear quantile(x,.10)', 'p25':'linear quantile(x,.25)',
 'p75':'linear quantile(x,.75)', 'p90':'linear quantile(x,.90)',
 'range':'max(x)-min(x)', 'iqr':'p75-p25', 'mad':'median(abs(x-median(x))), scale=1',
 'cv':'sample SD/mean; missing if mean=0', 'first':'earliest accepted in-window value',
 'latest':'latest accepted value strictly before cutoff',
 'slope':'OLS slope with intercept against (epoch-first_epoch)/60',
 'net_change':'last-first accepted value',
 'half_mean_change':'mean(second half)-mean(first half); each half n>=2/span>=1s',
 'skewness':'scipy.stats.skew(x,bias=False)',
 'excess_kurtosis':'scipy.stats.kurtosis(x,fisher=True,bias=False)',
 'mean_absolute_change':'mean(abs(diff)) over original valid adjacent pairs only',
 'median_absolute_change':'median(abs(diff)) over original valid adjacent pairs only',
 'rms_change':'sqrt(mean(diff^2)) over original valid adjacent pairs; NOT HRV',
 'observed_minute_mean':'equal mean of occupied UTC-minute means; no empty-minute filling',
 'observed_minute_sd':'sample SD of occupied UTC-minute means',
 'rolling5_mean':'mean of observed-bin means in eligible overlapping 5-minute blocks',
 'rolling5_min':'minimum qualifying 5-minute observed-bin mean',
 'rolling5_max':'maximum qualifying 5-minute observed-bin mean',
 'rolling5_within_sd':'mean sample SD of observed minute means within eligible blocks',
 'added_supported':'sum(nonnegative counter differences) over wholly contained supported pairs',
 'supported_rate':'60*sum(differences)/sum(supported seconds)',
 'median_pair_rate':'median(60*delta/dt) over supported pairs',
 'p90_pair_rate':'linear quantile(60*delta/dt,.9)',
 'max_pair_rate':'max(60*delta/dt)', 'sd_pair_rate':'sample SD(60*delta/dt)',
 'zero_change_supported_fraction':'sum(dt where delta=0)/sum(dt)',
}


def dictionary(cfg):
    rows = []
    for sensor in ['hr', 'pulseox', 'steps', 'floors']:
        for m in cfg['lookbacks_minutes']:
            for metric in STAT if sensor in ['hr', 'pulseox'] else ACTIVITY:
                base = 'bpm' if sensor == 'hr' else 'percent-like' if sensor == 'pulseox' else sensor
                units = (base+'^2' if metric == 'variance' else base+'/minute' if 'rate' in metric or
                         metric == 'slope' else 'dimensionless' if metric in ['cv','skewness',
                         'excess_kurtosis','zero_change_supported_fraction'] else base)
                req = 'n>=2; span>=1s'
                if metric in ['first', 'latest']:
                    req = 'n>=1'
                elif metric == 'slope':
                    req = 'n>=3; span>=60s'
                elif metric in ['skewness','excess_kurtosis']:
                    req = 'n>=8; span>=60s; nonconstant'
                elif 'absolute_change' in metric or metric == 'rms_change':
                    req = '>=2 original neighboring accepted pairs; each gap<=60s'
                elif metric.startswith('observed_minute'):
                    req = '>=2 occupied UTC minutes'
                elif metric.startswith('rolling5'):
                    req = '>=1 fully contained 5-minute block; >=3 observed minute bins per block'
                elif metric == 'half_mean_change':
                    req = 'each half n>=2; span>=1s'
                elif sensor in ['steps','floors']:
                    req = ('>=2' if metric == 'sd_pair_rate' else '>=1')+' fully contained supported counter pairs; dt<=300s'
                rows.append(dict(name=f'{sensor}__{metric}__{m}m', modality=sensor, metric=metric,
                    lookback_minutes=m, formula=FORMULAS[metric], units=units, requirements=req,
                    missing='NaN with explicit private status when support/validity fails',
                    source='n_epoch_s,n_value; accepted logical groups/counter-pair ledger',
                    status='provisional_pulseox_sensitivity' if sensor == 'pulseox' else 'supported_observed_summary',
                    limitation='Irregular recorded samples; acquisition and physiological validity unverified'))
    for m in cfg['lookbacks_minutes']:
        for key in ['hr_zero_step_mean','hr_positive_step_mean','hr_step_stratum_difference']:
            rows.append(dict(name=f'interaction__{key}__{m}m', modality='interaction',metric=key,
                lookback_minutes=m, units='bpm', formula='Mean HR in supported zero/positive step pairs; positive minus zero',
                requirements='each reported stratum >=3 HR readings spanning >=60s',missing='NaN if stratum absent',
                source='accepted HR and supported step-pair time overlap',status='supported_observed_summary',
                limitation='co-observation contrast; not causal adjustment or resting HR'))
    for m in cfg['lookbacks_minutes'][:-1]:
        for metric in ['mean','sd']:
            rows.append(dict(name=f'multiscale__adjacent_{metric}_difference__{m}m',modality='multiscale',
                metric=metric,lookback_minutes=m, units='bpm',formula=f'HR {metric} [c-L,c) minus [c-2L,c-L)',
                requirements='each non-overlapping interval n>=2/span>=1s',missing='NaN if either unsupported',
                source='accepted HR',status='supported_observed_summary',limitation='observed summaries, not full interval coverage'))
    for key in ['reference_mean_bpm','hr_mean_deviation_bpm','hr_latest_deviation_bpm']:
        rows.append(dict(name=key,modality='history',metric=key,lookback_minutes=60, units='bpm',
            formula='Earlier reference equal-weight UTC-minute mean [c-25h,c-1h); current mean/latest minus reference',
            requirements='>=30 occupied minutes and >=3 UTC hours; deviations need current summary',
            missing='NaN if history/current input unavailable',source='Notebook10 verified reference_features.parquet',
            status='supported_observed_summary',limitation='earlier recording, not resting; current/reference weighting differs'))
    return rows
