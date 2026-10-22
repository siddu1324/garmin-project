"""Synthetic timing, preservation and separation checks; no real records."""
import json

import numpy as np
import pandas as pd
import pytest

from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import publish_bundle, verify_bundle
from garmin_preprocessing.normalize import RAW, annotate_duplicates, normalize_chunk, parse_ema_clock


def test_ema_offset_and_precision():
    raw = pd.DataFrame({'actual_start_local': ['2030-01-01 04:01', '2030-01-01 04:01:30', 'bad'],
                        'scheduled_start_local': ['2030-01-01 04:00'] * 3,
                        'timezone_offset': ['-480'] * 3, 'instance_id': ['1893499200'] * 3})
    out = parse_ema_clock(raw)
    assert out.n_cutoff_epoch.iloc[0] == 1893499200
    assert out.n_cutoff_epoch.iloc[1] == 1893499290
    assert out.n_scheduled_epoch_residual_s.iloc[0] == 0
    assert out.n_invalid_clock.iloc[2] and np.isnan(out.n_cutoff_epoch.iloc[2])
    assert raw.actual_start_local.iloc[2] == 'bad'


def sensor_fixture():
    return pd.DataFrame({'ts': ['1893499199.25'] * 2, 'dte_tme': ['2030-01-01 03:59:59.250'] * 2,
                         'rsp_id': ['SYNTHETIC_R'] * 2, 'mbl_cod': ['SYNTHETIC_DEV'] * 2,
                         'data_type': ['ibi'] * 2, 'val': ['0800', '0800']})


def test_sensor_raw_values_rows_mapping_and_no_extra_shift():
    raw = sensor_fixture()
    original = raw.copy()
    config = {'clock_residual_tolerance_seconds': 1.1, 'allowed_offsets_minutes': [-480]}
    source = {'source_id': 'synthetic', 'content_sha256': 'synthetic-content', 'archive_sha256': 'synthetic-archive'}
    f = normalize_chunk(raw, source, 'SYNTHETIC_DEV', 3, {'SYNTHETIC_R': 'SYNTHETIC_DEV'}, set(), 101, config)
    assert f[RAW].equals(original)
    assert f.n_value.tolist() == [800, 800]
    assert f.n_epoch_s.iloc[0] == 1893499199.25
    assert f.n_inferred_offset_minutes.tolist() == [-480, -480]
    assert f.n_clock_residual_s.abs().lt(1e-6).all()
    assert f.n_source_row.tolist() == [101, 102]
    assert f.n_fold.eq(3).all() and f.n_partition.eq('development').all()
    with pytest.raises(AuditError):
        normalize_chunk(raw, source, 'SYNTHETIC_TEST', -1, {'SYNTHETIC_R': 'SYNTHETIC_DEV'}, set(), 1, config)


def test_duplicate_value_is_not_duplicate_beat_and_conflicts_are_retained():
    f = sensor_fixture()
    f = pd.concat([f, f.iloc[:1], f.iloc[:1]], ignore_index=True)
    f.loc[2, 'ts'] = '1893499200.25'  # Same IBI value at another time is legitimate.
    f.loc[3, 'val'] = '900'  # Distinct value at first timestamp is an unresolved conflict.
    f['n_source_columns'] = json.dumps(RAW)
    f['n_source_id'] = 'synthetic'
    f['n_source_row'] = np.arange(1, 5)
    f['n_epoch_s'], f['n_value'] = f.ts.astype(float), f.val.astype(float)
    result = annotate_duplicates(f)
    assert len(result) == 4 and result.n_logical_keep.sum() == 3
    assert result.q_exact_record_excess.sum() == 1
    assert not result.q_exact_record_duplicate.iloc[2]
    assert result.q_conflicting_timestamp_value.tolist() == [True, True, False, True]
    assert result.val.tolist() == f.val.tolist()


def test_immutable_bundle_reuse_and_no_overwrite(tmp_path):
    build = tmp_path / 'build'
    build.mkdir()
    (build / 'synthetic.txt').write_text('synthetic')
    destination = tmp_path / 'version'
    m = publish_bundle(build, destination, {'version': 'synthetic'})
    assert verify_bundle(destination) == m
    (destination / 'synthetic.txt').write_text('changed')
    with pytest.raises(AuditError):
        verify_bundle(destination)


def policy_fixture():
    from pathlib import Path
    from garmin_preprocessing.eligibility import parameter_array
    config = json.loads((Path(__file__).parents[1] / 'config/ibi_eligibility.json').read_text())
    return parameter_array(config['policies'][0])


def test_window_grid_and_strict_timestamp_membership():
    from garmin_preprocessing.eligibility import evaluate_windows, window_ends
    assert len(window_ends(10000, 5)) == 1
    assert len(window_ends(10000, 15)) == 11
    assert len(window_ends(10000, 1440)) == 1436
    assert window_ends(10000, 1440)[-1] - 300 == 10000 - 86400
    assert len(window_ends(np.nan)) == 0
    t = np.array([0., 300., 301.])
    z = np.zeros(3, dtype=bool)
    _, _, recorded = evaluate_windows(t, np.ones(3), z, z, z, np.array([300.]), 1, policy_fixture())
    assert recorded[0] == 1  # Left included; right and later timestamps excluded.


def test_interval_boundary_scenarios_are_not_assumed_equivalent():
    from garmin_preprocessing.eligibility import evaluate_windows
    t = np.array([.2, 100., 299.8])
    rr = np.full(3, .8)
    z = np.zeros(3, dtype=bool)
    _, end, _ = evaluate_windows(t, rr, z, z, z, np.array([300.]), 0, policy_fixture())
    _, start, _ = evaluate_windows(t, rr, z, z, z, np.array([300.]), 1, policy_fixture())
    assert end[0, 0] == start[0, 0] == 2
    assert end[0, 1] == start[0, 1] == 1
    assert end[0, 7] == 100. and start[0, 7] == .2
    assert np.isclose(end[0, 8], .2) and np.isclose(start[0, 8], 200.)


def test_ideal_support_passes_without_being_certified_normal():
    from garmin_preprocessing.eligibility import evaluate_windows
    t = np.arange(0., 300., .8)
    z = np.zeros(len(t), dtype=bool)
    mask, metrics, _ = evaluate_windows(t, np.full(len(t), .8), z, z, z, np.array([300.]), 0, policy_fixture())
    assert mask[0] == 0
    assert 298 < metrics[0, 3] < 300


def test_rejected_interval_does_not_create_artificial_adjacency():
    from garmin_preprocessing.eligibility import evaluate_windows
    t = np.arange(0., 300., .8)
    rr = np.full(len(t), .8)
    rr[len(t)//2] = .1
    z = np.zeros(len(t), dtype=bool)
    mask, metrics, _ = evaluate_windows(t, rr, z, z, z, np.array([300.]), 0, policy_fixture())
    assert mask[0] & 4
    assert mask[0] & 1024
    assert metrics[0, 4] < 151  # Two disconnected sequences, never a 299-second run.


def test_no_lookahead_artifact_flag_from_post_boundary_record():
    from garmin_preprocessing.eligibility import evaluate_windows
    t = np.arange(0., 300., .8)
    rr = np.full(len(t), .8)
    z = np.zeros(len(t), dtype=bool)
    first = evaluate_windows(t, rr, z, z, z, np.array([300.]), 0, policy_fixture())
    t2, rr2 = np.r_[t, 300.], np.r_[rr, 2.]
    z2 = np.zeros(len(t2), dtype=bool)
    second = evaluate_windows(t2, rr2, z2, z2, z2, np.array([300.]), 0, policy_fixture())
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_timestamp_conflicts_reject_instead_of_averaging():
    from garmin_preprocessing.eligibility import evaluate_windows
    t = np.arange(0., 300., .8)
    rr = np.full(len(t), .8)
    z = np.zeros(len(t), dtype=bool)
    conflict = z.copy()
    conflict[100] = True
    mask, _, _ = evaluate_windows(t, rr, conflict, conflict, z, np.array([300.]), 0, policy_fixture())
    assert mask[0] & 8 and mask[0] & 16
    assert np.all(rr == .8)


def test_unique_time_union_counts_overlap_once_and_preserves_gaps():
    from garmin_preprocessing.eligibility import support_seconds, union_seconds
    assert union_seconds(np.array([0., 60., 500.]), np.array([300., 360., 560.])) == 420
    # End-anchored starts need not be ordered; start-anchored ends need not be ordered.
    t, rr = np.array([1., 2., 3., 10.]), np.array([.5, 2., .5, 1.])
    chosen = np.ones(4, dtype=bool)
    for scenario in [0, 1]:
        starts, ends = (t - rr, t) if scenario == 0 else (t, t + rr)
        assert np.isclose(support_seconds(t, rr, chosen, scenario), union_seconds(starts, ends))


def test_privacy_numeric_aggregate_exception_is_schema_limited(tmp_path):
    from garmin_inventory.privacy import contains_private_token, public_scan_text
    path = tmp_path / 'attrition.json'
    row = dict(policy='screen', scenario='interval_end', lookback_minutes=5,
               survey_category='All', reason='clock_flag', windows=12345)
    path.write_text(json.dumps([row]))
    assert not contains_private_token(public_scan_text(path), ['12345'], [])
    row['windows'] = '12345'
    path.write_text(json.dumps([row]))
    assert contains_private_token(public_scan_text(path), ['12345'], [])
    row['windows'] = 12345
    row['participant'] = '12345'
    path.write_text(json.dumps([row]))
    assert contains_private_token(public_scan_text(path), ['12345'], [])


@pytest.mark.parametrize('offset', [-480, -420, -360, -300, 330])
def test_ema_offset_sign_is_row_specific(offset):
    raw = pd.DataFrame({'actual_start_local': ['2030-01-01 12:34:56'],
                        'scheduled_start_local': ['2030-01-01 12:30:00'],
                        'timezone_offset': [str(offset)], 'instance_id': ['0']})
    parsed = parse_ema_clock(raw)
    expected = pd.Timestamp('2030-01-01 12:34:56', tz='UTC') - pd.Timedelta(minutes=offset)
    assert parsed.n_actual_utc.iloc[0] == expected
    assert parsed.n_cutoff_utc.iloc[0] == expected


def test_disconnected_recordings_are_not_joined_across_a_gap():
    from garmin_preprocessing.eligibility import evaluate_windows
    t = np.r_[np.arange(0., 120., .8), np.arange(180., 300., .8)]
    rr = np.full(len(t), .8)
    z = np.zeros(len(t), dtype=bool)
    mask, metrics, _ = evaluate_windows(t, rr, z, z, z, np.array([300.]), 0, policy_fixture())
    assert mask[0] & 128 and mask[0] & 1024
    assert metrics[0, 4] < 120
    assert metrics[0, 3] < 240
    assert metrics[0, 11] > 60
