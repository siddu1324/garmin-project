"""Evidence boundaries and duplicate ambiguity, using synthetic records only."""

import pandas as pd
import pytest

from garmin_inventory.privacy import AuditError
from garmin_measurement_source.audit import export_shape, source_candidates, synthetic_ambiguity
from garmin_preprocessing.normalize import RAW, annotate_duplicates


def test_flat_schema_cannot_identify_sdk_batching_or_acquisition():
    result = export_shape([{'fields': RAW}])
    assert not result['sdk_or_version_identified']
    assert not result['acquisition_clock_verified']
    assert not result['derived_flags_are_device_annotations']
    with pytest.raises(AuditError):
        export_shape([{'fields': RAW + ['batch_index']}])


def test_source_search_preserves_candidate_without_reading_private_exports():
    found = source_candidates([
        '/project/collector/Listener.kt', '/project/app/build.gradle.kts',
        '/project/private/Listener.kt', '/project/node_modules/example.ts',
        '/project/data/export.swift', '/project/analysis.py',
    ])
    assert found['application_sources'] == ['/project/collector/Listener.kt']
    assert found['build_manifests'] == ['/project/app/build.gradle.kts']


def test_exact_row_collapse_is_reversible_but_does_not_prove_duplicate_beat():
    example = synthetic_ambiguity()
    f = pd.DataFrame({field: ['synthetic'] * 3 for field in RAW})
    f['ts'] = ['10.0', '10.0', '10.8']
    f['val'] = ['800'] * 3
    f['n_epoch_s'] = [10., 10., 10.8]
    f['n_value'] = [800.] * 3
    f['n_source_columns'] = 'synthetic schema'
    f['n_source_id'] = 'synthetic source'
    f['n_source_row'] = [1, 2, 3]
    result = annotate_duplicates(f)
    assert len(result) == 3  # All physical rows survive, including the ambiguous copy.
    assert result.n_logical_keep.tolist() == [True, False, True]
    assert result.val.tolist() == f.val.tolist()
    assert not example['distinguishable_from_export_alone']
    assert example['physical_rows_preserved'] == 2
