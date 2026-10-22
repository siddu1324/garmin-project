"""Assemble one executable notebook with byte-identical canonical source cells."""

from pathlib import Path
import nbformat
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import write_json
from .sources import SECTIONS, source_path


def build(root):
    root = Path(root)
    if (root / "reports/complete_pipeline_review/completion.json").exists():
        raise ValueError("Preserve completed review notebook")
    nb = nbformat.v4.new_notebook(
        metadata=dict(
            kernelspec=dict(name="python3", display_name="Python 3", language="python"),
            garmin_public_aggregate_notebook=True,
        )
    )
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    nb.cells = [
        md("# Complete Garmin + EMA pipeline"),
        code("""from pathlib import Path
import sys, json, importlib, hashlib
root = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(root / 'src'))
import numpy as np
import pandas as pd
from IPython.display import display, Image
from IPython.core.magic import register_cell_magic
from garmin_pipeline_review.sources import source_path
from garmin_pipeline_review.workflow import execute_review
from garmin_pipeline_review.audit import load

@register_cell_magic
def canonical_source(line, cell):
    path = source_path(root, line.strip())
    assert cell == path.read_text()
    compile(cell, str(path), 'exec')
    module = importlib.import_module(line.strip())
    assert Path(module.__file__).resolve() == path.resolve()
    return None

# Import exact source; do not rerun expensive extraction/search entrypoints.
public = root / 'reports/complete_pipeline_review'
def table(name):
    return pd.read_csv(public / (name + '.csv'))
def figure(name):
    display(Image(filename=str(public / 'figures' / (name + '.png'))))
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
"""),
        md("## Checkpoints and scope"),
        code("""display(pd.DataFrame([
    [13, '678 candidates; 438 supported + 240 provisional', 'recorded-time features; full logistic'],
    [14, '438 supported; PulseOx separate', 'continuous regression'],
    [15, '438 multiscale; 47 single-window', 'binary logistic'],
    [16, 'bounded subsets of the complete matrix', 'sampling and model diagnostics'],
    [17, 'past-only wearable references; EMA history gated', 'answer availability unverified'],
    [18, 'matched measurement/item diagnostics', 'no replacement of the complete matrix'],
    [19, 'reviewed-metadata importer and recall alignment', 'real aligned/history prediction blocked'],
], columns=['notebook', 'inputs', 'status']))
# Supported means computable observed summaries, not verified acquisition or physiology.
# No final-test data; no new feature definitions or model search.
"""),
    ]
    nb.cells += [
        md("## Frozen configurations"),
        code("""for name in ['comprehensive_features','comprehensive_regression','binary_6hour','binary_improvement','temporal_personalization','root_cause_analysis','measurement_alignment']:
    print(name)
    print(json.dumps(json.loads((root / 'config' / (name + '.json')).read_text()), indent=2))
"""),
    ]
    inventory = []
    for title, modules in SECTIONS:
        nb.cells.append(md("## " + title))
        for module in modules:
            path = source_path(root, module)
            text = path.read_text()
            # Read the complete current implementation before embedding it.
            compile(text, str(path), "exec")
            nb.cells.append(md("### " + str(path.relative_to(root))))
            nb.cells.append(code("%%canonical_source " + module + "\n" + text))
            inventory.append(
                dict(
                    module=module,
                    path=str(path.relative_to(root)),
                    sha256=sha256(path),
                    lines=len(text.splitlines()),
                )
            )
        if title == "PSS scoring, labels and context":
            nb.cells.append(
                code("""from garmin_ema_baseline.linkage import score_items
from garmin_pss_classification.data import targets
from garmin_preprocessing.normalize import parse_ema_clock
# Synthetic calculations only; stored item codes are 1–5.
synthetic = pd.DataFrame({'PSS_NERVOUS':['1','3','5',''], 'PSS_COPE':['5','3','1','4'], 'PSS_CONTROL':['5','4','1','4']})
scored = score_items(synthetic)
display(pd.DataFrame({'nervous':synthetic.PSS_NERVOUS, 'cope':synthetic.PSS_COPE, 'control':synthetic.PSS_CONTROL, 'composite':scored.pss_combo, 'three_class':targets(scored.pss_combo,'three_class'), 'at_least_5':targets(scored.pss_combo,'at_least_5'), 'at_least_10':targets(scored.pss_combo,'at_least_10')}))
# Exploratory labels, not clinical diagnoses.
clocks = pd.DataFrame({'actual_start_local':['01/01/2030 09:22','01/01/2030 09:22:35'], 'scheduled_start_local':['01/01/2030 09:20','01/01/2030 09:20'], 'timezone_offset':['-240','-240'], 'instance_id':['0','0']})
parsed = parse_ema_clock(clocks)
display(parsed[['n_actual_utc','n_start_precision','n_cutoff_utc']])
""")
            )
        if title == "Complete candidate feature generation":
            nb.cells.append(
                code("""from garmin_comprehensive.features import observed_features, activity_features
from garmin_comprehensive.augment import past_reference
cfg = json.loads((root / 'config/comprehensive_features.json').read_text())
# Synthetic HR: the rejected point remains a break for adjacent-change features.
synthetic_hr = pd.DataFrame({'n_epoch_s':[0.,60.,120.,180.,240.], 'n_value':[60.,70.,999.,80.,90.], 'a_acceptable':[True,True,False,True,True]})
values, quality, reasons, _ = observed_features(synthetic_hr,0.,300.,cfg)
display(pd.DataFrame([{'example':'synthetic HR','mean':values['mean'],'sample_sd':values['sd'],'slope_per_minute':values['slope'],'latest':values['latest'],'original_valid_pairs':quality['valid_adjacent_pairs'],'observed_minute_mean':values['observed_minute_mean']}]))
# Synthetic counter: rejected/reset/cross-boundary pairs do not contribute.
pairs = pd.DataFrame({'n_epoch_s':[30.,90.,150.,210.,270.], 'a_pair_start':[-30.,30.,90.,150.,210.], 'a_pair_delta':[10.,12.,-20.,0.,6.], 'a_pair_duration':[60.]*5, 'a_pair_accepted':[True,True,False,False,True], 'a_pair_rejection_mask':[0,0,128,2,0]})
activity, support, _ = activity_features(pairs,0.,300.)
display(pd.DataFrame([{**activity,**support}]))
""")
            )
    nb.cells += [
        md("## Recompute and verify saved artifacts"),
        code("""verification = execute_review(root)
display(pd.DataFrame([verification]))
cfg, frame, blocks, entries = load(root)
display(table('dataset_shapes'))
display(table('feature_counts'))
assert len(entries) == 678 and len(blocks['full']) == 438
assert len(blocks['pulseox_full']) - len(blocks['full']) == 240
assert frame.n_partition.eq('development').all()
"""),
        md("## All 678 feature definitions"),
        code("""# Full registry: formulas, sources, support requirements and evidence status.
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    display(table('feature_dictionary')[['name','modality','lookback_minutes','formula','requirements','status']])
"""),
        md("## Missingness and real-data calculation agreement"),
        code("""availability = table('feature_availability')
all_rows = availability[availability.survey.eq('All')]
display(all_rows.groupby('modality').agg(columns=('feature','size'), median_missing_fraction=('missing_fraction','median'), minimum_available_responses=('available','min'), maximum_available_responses=('available','max')))
display(table('feature_trace_checks'))
# Ten deterministic development responses; all 678 values checked; no participant rows displayed.
"""),
        md("## Actual encoded model inputs"),
        code("""display(table('transforms'))
# 438 candidate values + training-only missingness indicators + context; PulseOx excluded.
mapping = table('mapping')
with pd.option_context('display.max_rows', None):
    display(mapping[(mapping.checkpoint == 15) & (mapping.task == 'A') & (mapping.fold == 4)].drop(columns=['checkpoint','task','fold']))
display(table('encoded').groupby(['checkpoint','task','fold','kind']).size().unstack('kind'))
"""),
        md("## Frozen folds and class support"),
        code("""support = table('support')
display(support[support.view.isin(['full','multiscale_u'])].pivot(index=['checkpoint','task','fold','role'],columns='label',values='responses').fillna(0).astype(int))
# Training rows recur across folds; validation support pools each response once.
"""),
        md("## Saved hyperparameter choices"),
        code("""display(table('settings'))
# Selected settings come from grouped inner validation, never held-out outcomes.
"""),
        md("## Three-class and binary classification"),
        code("""classification = table('classification_metrics')
display(classification[['checkpoint','task','view','responses','participants','roc_auc','macro_f1','balanced_accuracy','elevated_precision','elevated_recall']])
figure('classification')
figure('confusion')
figure('binary')
display(table('per_class'))
# Three-class AUROC is macro one-vs-rest. Binary AUROC uses the positive class.
"""),
        md("## Matched regression results"),
        code("""regression = table('regression_metrics')
display(regression[['checkpoint','view','family','responses','participants','mae','rmse','r2','spearman']])
figure('regression')
figure('score_bands')
display(table('regression_score_bands'))
# Within each task, full/context models use identical 3,689 validation responses.
# Repeated exploratory comparisons do not establish a deployable or causal physiological signal.
"""),
        md("## Current measurement gates"),
        code("""checkpoint = json.loads((root / 'reports/measurement_alignment/completion.json').read_text())
display(pd.DataFrame({'status':['BLOCKED','BLOCKED'], 'reason':checkpoint['blocked']}))
# Raw clocks, final-test sealing, earlier artifacts and frozen folds remain unchanged.
"""),
        md("## Verification"),
        code("""from garmin_pipeline_review.verify import verify
checks = verify(root)
display(pd.DataFrame([checks]))
"""),
    ]
    nbformat.write(nb, root / "notebooks/20_complete_pipeline_for_review.ipynb")
    write_json(root / "reports/complete_pipeline_review/source_inventory.json", inventory)
    return dict(
        canonical_modules=len(inventory),
        canonical_lines=sum(r["lines"] for r in inventory),
        code_cells=sum(c.cell_type == "code" for c in nb.cells),
    )


if __name__ == "__main__":
    print(build(Path.cwd()))
