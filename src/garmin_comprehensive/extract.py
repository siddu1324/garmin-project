"""Partitioned extraction from immutable development-only inputs."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import check_development
from garmin_hr_activity.features import prepare, empty_frame
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (digest_object, frozen_assignments, input_guard,
                                        publish_bundle, verify_bundle, write_json)
from garmin_pss_classification.data import load_development, HISTORY
from .dictionary import dictionary
from .features import observed_features, activity_features, coobserved

REFERENCE = Path('/Users/siddharthareddy/SEA-RESEARCH/HEALTH-WEARABLE-PROJECT/Analysis/sea-garmin')


def extract(root):
    root = Path(root)
    cfg = json.loads((root/'config/comprehensive_features.json').read_text())
    input_guard(root, REFERENCE, root/'data')
    assignments, dev = frozen_assignments(root, REFERENCE)
    frame = load_development(root, cfg, assignments)
    normalized, old = root/cfg['normalized_bundle'], root/cfg['feature_bundle']
    verify_bundle(normalized)
    old_features = pd.read_parquet(old/'features.parquet').set_index('n_ema_id')
    meta = pd.read_parquet(normalized/'ema/development_ema.parquet',
                           columns=['n_ema_id','n_cutoff_epoch','n_invalid_clock','n_start_precision'])
    frame = frame.merge(meta, on='n_ema_id', validate='one_to_one')
    inputs = dict(config=cfg, specification_sha256=sha256(root/'reports/comprehensive_features/specification.md'),
        bundles={k:sha256(root/cfg[k]/'manifest.json') for k in cfg if k.endswith('_bundle')},
        assignments=sha256(root/'private/frozen_participant_assignments.parquet'),
        environment=sha256(root/'uv.lock'), code={n:sha256(Path(__file__).with_name(n))
            for n in ['extract.py','features.py','dictionary.py']})
    version = digest_object(inputs)[:16]
    dest = root/'private/comprehensive_features/features'/version
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        verify_bundle(dest)
        return dest
    build = Path(tempfile.mkdtemp(prefix='.building_',dir=dest.parent))
    entries = dictionary(cfg)
    people = json.loads((normalized/'participant_directories.json').read_text())
    assert {p['participant'] for p in people} == dev
    values, quality, lineage = [], [], []
    legacy_checks = 0
    source_hashes = {}
    count_input = dict(hr=0,steps=0,floors=0,pulseox=0)
    hr_cfg = json.loads((root/'config/hr_activity.json').read_text())
    for num, person in enumerate(people):
        e = frame[frame.n_participant.eq(person['participant'])]
        streams = {}
        for sensor, raw in [('hr','hr'),('steps','steps'),('floors','floorsClimbed'),('pulseox','pulseOx')]:
            if sensor == 'hr':
                path = old/'reading_dispositions'/person['directory']/'hr.parquet'
                f = pd.read_parquet(path)
                if len(f):
                    assert f.n_participant.eq(person['participant']).all() and f.n_partition.eq('development').all()
                    assert f.n_fold.eq(person['fold']).all()
                g = f[f.a_first & np.isfinite(f.n_epoch_s)].copy() if len(f) else prepare(empty_frame(),'hr',hr_cfg)[0]
            elif sensor in ['steps','floors']:
                path = old/'counter_pairs'/person['directory']/f'{raw}.parquet'
                g = pd.read_parquet(path)
            else:
                path = normalized/'logical'/person['directory']/'pulseOx.parquet'
                f = pd.read_parquet(path) if path.exists() else empty_frame()
                if len(f):
                    assert f.n_participant.eq(person['participant']).all() and f.n_partition.eq('development').all()
                    assert f.n_fold.eq(person['fold']).all()
                bounds = cfg['pulseox_provisional_range']
                g, _ = prepare(f,'hr',{**hr_cfg,'hr_min_bpm':bounds[0],'hr_max_bpm':bounds[1]})
            source_hashes[str(path.relative_to(root))] = sha256(path) if path.exists() else None
            count_input[sensor] += len(g)
            streams[sensor] = g.reset_index(drop=True)
        for row in e.itertuples(index=False):
            output = {'n_ema_id':row.n_ema_id}
            for m in cfg['lookbacks_minutes']:
                end = float(row.n_cutoff_epoch)
                if row.n_invalid_clock or not np.isfinite(end):
                    continue
                start = end-60*int(m)
                for sensor, g in streams.items():
                    if sensor in ['hr','pulseox']:
                        f, q, reasons, bounds = observed_features(g,start,end,cfg)
                    else:
                        f, q, bounds = activity_features(g,start,end)
                        reasons = {k:'available' if np.isfinite(v) else
                            'no_recording' if not q['recorded_groups'] else 'no_supported_pair_or_insufficient_pairs' for k,v in f.items()}
                    output.update({f'{sensor}__{k}__{m}m':v for k,v in f.items()})
                    quality.append(dict(n_ema_id=row.n_ema_id,sensor=sensor,lookback_minutes=m,
                                        **q,reasons_json=json.dumps(reasons,sort_keys=True)))
                    lineage.append(dict(n_ema_id=row.n_ema_id,sensor=sensor,lookback_minutes=m,
                        start_epoch=start,end_epoch=end,directory=person['directory'],group_start=bounds[0],group_stop=bounds[1]))
                output.update({f'interaction__{k}__{m}m':v for k,v in
                               coobserved(streams['hr'],streams['steps'],start,end).items()})
                if m < 1440:
                    earlier, _, _, _ = observed_features(streams['hr'],start-60*int(m),start,cfg)
                    for metric in ['mean','sd']:
                        output[f'multiscale__adjacent_{metric}_difference__{m}m'] = output[f'hr__{metric}__{m}m']-earlier[metric]
                # Independent equality against frozen old implementation/output, not a new expected value.
                mapping = {'mean':'mean_bpm','median':'median_bpm','sd':'sd_bpm','minimum':'minimum_bpm',
                           'maximum':'maximum_bpm','p25':'q25_bpm','p75':'q75_bpm','latest':'latest_bpm',
                           'slope':'slope_bpm_per_minute'}
                for a,b in mapping.items():
                    if not np.isclose(output[f'hr__{a}__{m}m'],old_features.loc[row.n_ema_id,f'hr_{b}__{m}m'],equal_nan=True):
                        raise AuditError('Expanded HR disagrees with frozen original summaries; details private.')
                    legacy_checks += 1
                for s, old_s in [('steps','steps'),('floors','floorsClimbed')]:
                    assert np.isclose(output[f'{s}__added_supported__{m}m'],old_features.loc[row.n_ema_id,f'{old_s}_added_supported__{m}m'],equal_nan=True)
                    legacy_checks += 1
            values.append(output)
        if (num+1)%10 == 0:
            print(f'Extracted {num+1}/{len(people)} development partitions',flush=True)
    generated = pd.DataFrame(values)
    names = [d['name'] for d in entries]
    base_columns = ['n_ema_id','n_participant','n_partition','n_fold','n_survey_category',
                    'n_cutoff_epoch','n_invalid_clock','n_start_precision','pss_combo','context_available',
                    'local_time_sin','local_time_cos', *HISTORY,
                    'target_three_class','target_at_least_5','target_at_least_10']
    data = frame[base_columns].merge(generated,on='n_ema_id',validate='one_to_one')
    for name in names:
        if name not in data:
            data[name] = np.nan
    data = data.sort_values('n_ema_id').reset_index(drop=True)
    check_development(data,assignments,3706)
    assert not np.isinf(data[names].to_numpy(float)).any()
    q = pd.DataFrame(quality)
    data.to_parquet(build/'development_features.parquet',index=False)
    q.to_parquet(build/'quality.parquet',index=False)
    pd.DataFrame(lineage).to_parquet(build/'lineage.parquet',index=False)
    write_json(build/'dictionary.json',entries)
    write_json(build/'source_partitions.json',source_hashes)
    coverage = []
    for d in entries:
        for survey in ['All','Morning','Midday','Endday']:
            f = data if survey == 'All' else data[data.n_survey_category.eq(survey)]
            ok = f[d['name']].notna()
            coverage.append(dict(feature=d['name'],modality=d['modality'],lookback_minutes=d['lookback_minutes'],
                survey=survey,responses=len(f),available=int(ok.sum()),missing=int((~ok).sum()),
                available_participants=f.loc[ok,'n_participant'].nunique(),missing_fraction=float((~ok).mean())))
    write_json(build/'coverage.json',coverage)
    write_json(build/'summary.json',dict(responses=len(data),participants=data.n_participant.nunique(),
        valid_scores=int(data.pss_combo.notna().sum()),feature_count=len(names),
        count_by_modality=pd.Series([d['modality'] for d in entries]).value_counts().to_dict(),
        count_by_lookback=pd.Series([str(d['lookback_minutes']) for d in entries]).value_counts().to_dict(),
        original_summary_agreement_checks=legacy_checks,input_group_counts=count_input,
        entirely_missing=[c for c in names if not data[c].notna().any()],
        complete_all_supported=int(data[[d['name'] for d in entries if d['modality']!='pulseox']].notna().all(axis=1).sum())))
    publish_bundle(build,dest,dict(inputs=inputs))
    return dest


if __name__ == '__main__':
    print(extract(Path.cwd()))
