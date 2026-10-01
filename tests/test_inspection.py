"""Synthetic tests for privacy, temporal boundaries, identity and test-set isolation."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from garmin_inventory.audit import (
    continuity, cutoff, ema_audit, history, resolve_pair, sensor_audit,
    source_identity_scan, validate_split,
)
from garmin_inventory.privacy import AuditError, scan_public
from garmin_inventory.sources import Source, safe_members


def test_cutoff_preserves_precision_policy():
    assert cutoff(1000, 'minute') == 940
    assert cutoff(1000, 'second') == 1000
    with pytest.raises(AuditError):
        cutoff(1000, 'unknown')


def test_history_strict_right_boundary_and_no_zero_fill():
    assert history([1000, 1001], 1000, 5, 60) == (False, False)
    assert history([700], 1000, 5, 60) == (True, False)
    assert history([], 1000, 5, 60) == (False, False)
    assert history(np.arange(700, 1000, 60), 1000, 5, 60) == (True, True)
    assert history([700, 999], 1000, 5, 60) == (True, False)
    # Internal coverage alone does not suffice when boundary support is absent.
    assert history([900, 950, 999], 1000, 5, 60) == (True, False)


def test_continuity_cannot_bridge_gaps_or_timestamp_duplicates():
    assert continuity([1, 1, 2, 100, 101], 2).tolist() == [1, 1]
    assert continuity([1], 2).tolist() == [0]
    assert continuity([], 2).tolist() == []


def assignment_fixture():
    return pd.DataFrame({'participant': [f'SYNTHETIC_{i:03}' for i in range(157)],
                         'partition': ['development'] * 125 + ['final_test'] * 32,
                         'fold': [i // 25 for i in range(125)] + [-1] * 32})


def test_split_requires_exact_cohort_and_folds():
    a = assignment_fixture()
    people = set(a.participant)
    assert len(validate_split(a, people)) == 125
    assert not set(a.loc[a.partition == 'final_test', 'participant']) & validate_split(a, people)
    with pytest.raises(AuditError):
        validate_split(pd.concat([a, a.iloc[:1]]), people)
    with pytest.raises(AuditError):
        validate_split(a, people | {'SYNTHETIC_NEW'})
    a.loc[0, 'fold'] = 4
    with pytest.raises(AuditError):
        validate_split(a, people)


def test_ambiguous_respondents_and_conflicts_are_not_forced():
    lookup = {'SYNTHETIC_R': 'SYNTHETIC_A'}
    assert resolve_pair('', 'SYNTHETIC_R', lookup, set())[0] == 'SYNTHETIC_A'
    assert resolve_pair('SYNTHETIC_B', 'SYNTHETIC_R', lookup, set()) == (None, 'conflicting')
    assert resolve_pair('', 'SYNTHETIC_R', lookup, {'SYNTHETIC_R'}) == (None, 'unresolved')
    assert resolve_pair('SYNTHETIC_A', 'SYNTHETIC_R', {}, {'SYNTHETIC_R'})[0] == 'SYNTHETIC_A'


def test_ema_outcome_and_clock_audits_exclude_final_test():
    raw = pd.DataFrame({
        'mbl_cod': ['SYNTHETIC_DEV', 'SYNTHETIC_TEST'], 'rsp_id': ['R_DEV', 'R_TEST'],
        'instance_id': ['1893499200', 'not an epoch'],
        'scheduled_start_local': ['2030-01-01 04:00:00', 'invalid'],
        'timezone_offset': ['-480', 'invalid'],
        'actual_start_local': ['2030-01-01 04:01', 'invalid'],
        'PSS_NERVOUS': ['1', '999'], 'PSS_COPE': ['5', '999'], 'PSS_CONTROL': ['3', '999'],
    })
    ema = raw.drop(columns=['PSS_NERVOUS', 'PSS_COPE', 'PSS_CONTROL']).assign(category='Morning')
    frame, clocks, dictionary, items = ema_audit(ema, [('Morning', raw)], {'SYNTHETIC_DEV'})
    assert len(frame) == 1 and clocks['actual_unparsed'] == 0
    assert clocks['scheduled_epoch_agreement_rows'] == 1
    assert items[0]['complete_valid_items'] == 1 and items[0]['nonmissing_invalid_cells'] == 0
    assert all(d['development_cells'] == 1 for d in dictionary)
    assert not any('score' in c for c in frame)


def make_source(tmp_path, name, content):
    path = tmp_path / name
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('synthetic.csv', content)
    return Source(path, 'synthetic.csv')


def test_sensor_audit_never_opens_final_test_source(tmp_path):
    source = make_source(tmp_path, 'synthetic.zip',
        'ts,dte_tme,mbl_cod,rsp_id,data_type,val\n'
        '1893499199,2030-01-01 11:59:59,SYNTHETIC_DEV,R_DEV,ibi,800\n'
        '1893499200,2030-01-01 12:00:00,SYNTHETIC_DEV,R_DEV,ibi,999\n')
    meta = [{'source': source, 'participants': {'SYNTHETIC_DEV'}, 'duplicate_export': False, 'unresolved_rows': 0},
            {'source': Source(tmp_path / 'must_not_open.zip', 'none'),
             'participants': {'SYNTHETIC_TEST'}, 'duplicate_export': False, 'unresolved_rows': 0}]
    ema = pd.DataFrame({'mbl_cod': ['SYNTHETIC_DEV'], 'cutoff': [1893499200.]})
    sensors, coverage, clocks, ibi, n = sensor_audit(meta, {'SYNTHETIC_DEV'}, ema)
    assert n == 1 and clocks['rows'] == 2
    assert next(s for s in sensors if s['sensor'] == 'ibi')['rows'] == 2
    assert next(c for c in coverage if c['sensor'] == 'ibi')['responses_with_any_observation'] == 1


def test_duplicate_export_accounting_uses_content(tmp_path):
    text = ('ts,dte_tme,mbl_cod,rsp_id,data_type,val\n'
            '1893499199,2030-01-01 11:59:59,SYNTHETIC_A,SYNTHETIC_R,ibi,800\n')
    sources = [make_source(tmp_path, f'synthetic_{i}.zip', text) for i in range(2)]
    ema = pd.DataFrame({'mbl_cod': ['SYNTHETIC_A'], 'rsp_id': ['SYNTHETIC_R']})
    _, summary, _ = source_identity_scan(sources, ema, tmp_path / 'unavailable_reference')
    assert summary['physical_rows'] == 2 and summary['logical_rows'] == 1
    assert summary['duplicate_export_files'] == 1


@pytest.mark.parametrize('name', ['../outside.csv', '/absolute.csv', 'bad\\name.csv'])
def test_archive_paths_are_checked_without_extraction(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr(name, 'synthetic')
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as z, pytest.raises(AuditError):
        safe_members(z)


def test_public_scan_rejects_private_tokens_and_notebook_output(tmp_path):
    (tmp_path / 'reports').mkdir()
    p = tmp_path / 'reports/report.md'
    p.write_text('A harmless aggregate count: 125')
    assert scan_public(tmp_path, {'SYNTHETIC_SECRET'}, set()) == 1
    p.write_text('SYNTHETIC_SECRET')
    with pytest.raises(AuditError):
        scan_public(tmp_path, {'SYNTHETIC_SECRET'}, set())
    p.unlink()
    (tmp_path / 'notebooks').mkdir()
    (tmp_path / 'notebooks/example.ipynb').write_text(
        '{"cells":[{"cell_type":"code","outputs":[{"text":"private"}],"execution_count":1}]}')
    with pytest.raises(AuditError):
        scan_public(tmp_path, set(), set())


def test_unavailable_split_completes_structure_without_exploration(tmp_path, monkeypatch):
    from garmin_inventory import audit
    data = tmp_path / 'data'
    data.mkdir()
    make_source(data, 'ema.zip',
        'mbl_cod,rsp_id,instance_id,scheduled_start_local,timezone_offset,actual_start_local\n'
        'SYNTHETIC_A,SYNTHETIC_R,1893499200,2030-01-01 12:00:00,0,2030-01-01 12:01:00\n')
    make_source(data, 'sensor.zip',
        'ts,dte_tme,mbl_cod,rsp_id,data_type,val\n'
        '1893499199,2030-01-01 11:59:59,SYNTHETIC_A,SYNTHETIC_R,ibi,800\n')
    monkeypatch.setattr(audit, 'git_guard', lambda root: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Exploration must not run without frozen assignments.')
    monkeypatch.setattr(audit, 'sensor_audit', forbidden)
    monkeypatch.setattr(audit, 'ema_audit', forbidden)
    report = audit.run(tmp_path, reference=tmp_path / 'absent')
    assert report['structural']['ema_rows'] == 1
    assert report['structural']['garmin']['logical_rows'] == 1
    assert report['frozen_split']['available'] is False
    assert 'sensors_development' not in report
    assert 'pss_item_validity_development' not in report


def test_git_guard_stops_on_staged_sensitive_file(tmp_path):
    import subprocess
    from garmin_inventory.privacy import git_guard
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'synthetic.csv').write_text('SYNTHETIC_PRIVATE\n')
    subprocess.run(['git', 'add', 'synthetic.csv'], cwd=tmp_path, check=True)
    with pytest.raises(AuditError, match='Tracked/staged'):
        git_guard(tmp_path)
    # The safeguard must not untrack anything automatically.
    assert subprocess.check_output(['git', 'ls-files'], cwd=tmp_path, text=True).strip() == 'synthetic.csv'


def test_identifier_scan_does_not_confuse_hash_fragments_with_keys():
    from garmin_inventory.privacy import contains_private_token
    assert not contains_private_token('sha256:abcdef12345abcdef', {'12345'}, set())
    assert contains_private_token('participant="12345"', {'12345'}, set())
    assert contains_private_token('person_12345.csv', {'12345'}, set())
    assert contains_private_token('path/synthetic-private.csv', set(), {'synthetic-private.csv'})
