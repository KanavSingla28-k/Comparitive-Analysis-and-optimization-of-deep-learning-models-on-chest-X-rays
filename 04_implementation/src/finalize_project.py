"""Phases 5-8 assembly. Run all or verify-package; no model training here."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime
from runtime import PROJECT_ROOT
from settings import CONFIG, PROTOCOL_ID
from review_statistics import review, read, save, csv_out

RESULTS=PROJECT_ROOT/'06_results'
REPORT=PROJECT_ROOT/'07_report'
HANDOFF=PROJECT_ROOT/'08_handoff'
PRIMARY=RESULTS/'retrained_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(c) for c in row)+' |' for row in rows])


def details():
    fields={}
    for line in (PROJECT_ROOT/'01_planning/PROJECT_DETAILS.md').read_text(encoding='utf-8').splitlines():
        if line.startswith('|'):
            cells=[c.strip() for c in line.split('|')[1:-1]]
            if len(cells)==2:
                fields[cells[0]]=cells[1]
    return fields


def literature():
    records=read(PROJECT_ROOT/'02_literature/verified_sources.json')['records']
    csv_out(PROJECT_ROOT/'02_literature/verified_sources.csv',[{'id':r['id'],'authors':'; '.join(r['authors']),'title':r['title'],'date':r['date'],'url':r['url'],'doi':r['doi'] or '', 'description':r['description']} for r in records])
    paragraphs=[]
    for r in records:
        author=r['authors'][0].split(',')[0]+' et al.'
        paragraphs.append(f"- [{r['id']}] {author} ({r['date'][:4]}). [{r['title']}]({r['url']}). {r['description']}")
    references='\n\n'.join(paragraphs)
    write(PROJECT_ROOT/'02_literature/VERIFIED_SOURCES.md','# Verified primary sources\n\nChecked on 22 October 2026. Literature scores are not local measurements.\n\n'+references+'\n\nThe MedMNIST paper was published in 2023 despite the DOI containing 2022. The actual 224-pixel file is separately identified by the official MedMNIST+ release and Phase 1 checksum, rather than being an upscaled 28-pixel image array.\n')
    return records,references


def figures(hardware):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    target=REPORT/'figures'
    target.mkdir(parents=True,exist_ok=True)
    pairs=read(PROJECT_ROOT/'05_experiments/retrained_v1/artifacts/statistical_summary.json')['paired_comparisons']
    fig,ax=plt.subplots(figsize=(10.5,4.2))
    labels=['ResNet partial minus frozen','ResNet partial minus DenseNet partial','MobileNet partial minus DenseNet partial']
    for i,p in enumerate(pairs):
        mean=p['delta_auroc_mean']; lo,hi=p['ci95']
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color='#2563eb')
    ax.axvline(0,color='#64748b',linestyle='--')
    ax.set(yticks=range(3),yticklabels=labels,xlabel='Mean AUROC difference with exploratory paired 95% interval',title='Paired comparisons: 3 seeds and 621 exact-image groups')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.15)
    fig.tight_layout(); fig.savefig(target/'paired_auroc_differences.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5)); x=np.arange(3)
    for k,(session,label) in enumerate([('reproduced_v1','Pilot session'),('retrained_v1','Fresh session')]):
        rows=[next(r for r in hardware if r['session']==session and r['architecture']==n) for n in CONFIG['architectures']]
        offset=-.17 if k==0 else .17
        axes[0].bar(x+offset,[r['synthetic_batch16_amp_speedup'] for r in rows],width=.32,label=label)
        axes[1].bar(x+offset,[r['synthetic_batch16_allocation_reduction_percent'] for r in rows],width=.32,label=label)
    axes[0].axhline(1,color='#64748b',linestyle='--')
    axes[0].set(ylabel='FP32 step time / AMP step time',title='Synthetic batch-16 speed ratios varied')
    axes[1].set(ylabel='Peak tensor allocation reduction (%)',title='AMP reduced allocation in both sessions')
    for ax in axes:
        ax.set_xticks(x,['ResNet-18','DenseNet-121','MobileNetV3\nSmall']); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.15)
    fig.suptitle('Short full-gradient probes: technical samples are not independent training runs',fontsize=10)
    fig.tight_layout(); fig.savefig(target/'session_precision_comparison.png',dpi=180,bbox_inches='tight'); plt.close(fig)


def optimization(hardware,precision):
    pair=read(PROJECT_ROOT/'05_experiments/retrained_v1/artifacts/statistical_summary.json')['paired_comparisons'][0]
    hardware_table=table(['Session','Architecture','Synthetic AMP speedup','Allocation reduction','AMP batch-64 MiB'],[[r['session'],r['architecture'],f"{r['synthetic_batch16_amp_speedup']:.3f}x",f"{r['synthetic_batch16_allocation_reduction_percent']:.1f}%",f"{r['amp_batch64_peak_mib']:.1f}"] for r in hardware])
    fresh=[r for r in precision if r['session']=='retrained_v1']
    precision_table=table(['Configuration','Max probability change','Threshold changes / 1872 seed predictions','Max AUROC change'],[[r['group'],f"{r['maximum_probability_difference']:.6f}",r['threshold_disagreements_over_1872_seed_predictions'],f"{r['maximum_absolute_auroc_difference']:.6f}"] for r in fresh])
    text=f'''# Phase 7 — measured optimization proof of concept

Partial ResNet fine-tuning changed mean test AUROC by **{pair['delta_auroc_mean']:+.6f}**, with exploratory paired 95% interval **[{pair['ci95'][0]:+.6f}, {pair['ci95'][1]:+.6f}]**. Both branches start from the same epoch-2 state and use the same seeds, image order, rotation draws, additional epochs and adaptation learning rate. The trainable modules differ. This supports a within-benchmark learning improvement, not compression or clinical validation.

## Hardware evidence

{hardware_table}

Above-one speedup means AMP was faster. Ten-step technical measurements varied substantially across the two sessions, including MobileNet's direction. They do not establish a stable model-specific acceleration or an independent run-to-run confidence interval. Allocation fell in both sessions, supporting an AMP memory-efficiency direction for these workloads.

All three models completed tested full-gradient AMP batch 64. DenseNet used approximately 4.17 GiB of allocated tensors, leaving less practical headroom on this 6 GB GPU. Driver allocations, unused reservations, data residency and long-run behavior are additional. These are feasible tested workloads, not absolute maximums. Batch 16 remains the validated real-training setting.

FP32 was faster for isolated batch-1 inference in both sessions. Timing excludes image decoding, preprocessing and transfer. Timing inputs were fresh random tensors of matching shape, not identical values; numerical consistency below used the SAME real test images and checkpoints. This timing limitation remains disclosed.

## Precision consistency

{precision_table}

MobileNet's probability changes and eight changed decisions across 1872 seed predictions show why AMP cannot be called numerically identical or automatically accuracy-preserving. Do not tune thresholds on these test differences. Training precision, isolated inference precision and predictive fine-tuning are separate choices.

## Next experiment

[The batch-32 validation pilot](../05_experiments/NEXT_OPTIMIZATION_EXPERIMENT.md) is a proposed next experiment, not an executed result. There is no measured theoretical optimization ceiling or guarantee that a batch change improves real training.

Phase 7 is technically complete through the paired learning signal and disclosed memory/precision probes. Quantization, pruning, distillation and architecture search remain future work.
'''
    write(RESULTS/'PHASE_7_OPTIMIZATION.md',text)
    save(RESULTS/'statistics/phase_7_acceptance.json',{'status':'PASS','protocol_id':PROTOCOL_ID,'description':'Paired fine-tuning signal and hardware proof accepted with session variability and timing-input limitations disclosed. Technical acceptance is not supervisor approval.','paired_learning':{**pair,'description':'Exploratory within-benchmark AUROC improvement, not compression.'},'hardware_sessions':hardware,'precision_consistency':fresh,'next_experiment_executed':False})

def final_report(records,references):
    original=(PRIMARY/'comparative_analysis_report.md').read_text(encoding='utf-8')
    fields=details()
    header='# Comparative Analysis and Optimization of Deep Learning Models on Chest X-rays\n\n**A reproducible three-model comparison for pediatric pneumonia classification**\n\n'
    keys=['College / department','Course name','Team member names','Roll / registration numbers','Supervisor / guide']
    header+=table(['Project detail','Recorded information'],[[k,fields.get(k,'Not supplied')] for k in keys])+'\n\n'
    header+='Technical scope: PneumoniaMNIST+ 224, three pretrained CNNs, three seeds and a five-epoch recipe. Team details are copied as supplied without inferring a name-to-roll-number mapping. Primary results below use the **fresh retrained_v1 session**. The pilot reproduced the same predictive results; its timing values are kept separate. This technical report does not invent institution-specific formatting approval or a supervisor signature.\n\n'
    body=original.split('\n',1)[1]
    related='''## 2. Related work

Wang et al. introduced report-derived labels for a chest radiograph benchmark. Distinguish the original eight-label release from later fourteen-label updates. [ChestX-ray8](https://arxiv.org/abs/1705.02315)

CheXNet motivates a DenseNet-121 baseline, but its radiologist F1 comparison has a specific test and annotation setting. It does not establish general clinical superiority. [CheXNet](https://arxiv.org/abs/1711.05225)

CheXpert represents uncertainty in report-derived observation labels and evaluates selected findings with experts. Our pediatric binary task and data differ. [CheXpert](https://arxiv.org/abs/1901.07031)

Bressem et al.'s sixteen-architecture study motivates measuring resource costs alongside predictive performance. Its dataset-specific scores are not our measurements. [Architecture comparison](https://www.nature.com/articles/s41598-020-70479-z)

Ke et al. studied transfer and truncation on CheXpert. A nonsignificant drop does not prove equivalence, and our fine-tuning control does not reproduce their truncation experiment. [CheXtransfer](https://arxiv.org/abs/2101.06871)

Yang et al.'s 2023 MedMNIST v2 paper documents the original standardized collection. Our actual 224-pixel MedMNIST+ file is separately identified by its release and checksum. [MedMNIST v2](https://www.nature.com/articles/s41597-022-01721-8), [actual release](https://zenodo.org/records/10519652)

'''
    body=re.sub(r'## 2\. Related work\n.*?(?=## 3\.)',lambda _:related,body,flags=re.S)
    body=body.replace('Provenance (imported evidence or new training) is recorded in provenance.json beside this report.','Fresh-training provenance is linked in the reproduction appendix.')
    body=body.replace('See `README.md` for commands.','See [run instructions](../04_implementation/README.md) for commands.')
    appendix='''
## 11. Independent acceptance and reproduction

Independent review passed 894 numerical checks across both sessions. A weighted positive-negative pair implementation reconstructed all 2000 bootstrap draws within absolute tolerance 1e-14. Raw timing medians and percentiles were verified. Inference wall-throughput denominators were not separately retained; the rates are recorded measurements with checked aggregation, not reconstructed event-time rates.

![Declared paired AUROC intervals](figures/paired_auroc_differences.png)

![Precision probes across both sessions](figures/session_precision_comparison.png)

All twelve trained checkpoint files, real test-prediction arrays and predictive bootstrap statistics reproduced exactly on this laptop. This is not a cross-platform bitwise promise. Short hardware speed ratios varied substantially, including MobileNet's direction; they are engineering probes, not guaranteed end-to-end acceleration.

- [Python settings](../04_implementation/src/settings.py), [training](../04_implementation/src/experiment.py), [analysis](../04_implementation/src/analyze.py), [independent reviewer](../04_implementation/src/review_statistics.py)
- [Fresh provenance](../05_experiments/retrained_v1/provenance.json), [execution log](../05_experiments/retrained_v1/execution.log), [environment](../05_experiments/retrained_v1/artifacts/environment.json)
- [Accepted comparison CSV](tables/accepted_comparison.csv), [per-seed scores](tables/metrics.csv), [epoch histories](tables/epoch_history.csv), [paired differences](tables/paired_differences.csv)
- [Precision comparison](tables/precision_consistency.csv), [hardware sessions](tables/hardware_sessions.csv), [optimization note](../06_results/PHASE_7_OPTIMIZATION.md)
- [Final JSON with numeric descriptions](final_results_explained.json), [run instructions](../04_implementation/README.md), [evidence index](../08_handoff/EVIDENCE_INDEX.md)

From the project root, `./run.ps1 reproduce-results -RunDir '05_experiments/demo_new'` replays checkpoints and rebuilds analysis. Use a new directory. `./run.ps1 full -RunDir '05_experiments/training_new'` retrains everything. `./finalize.ps1 verify-package` checks final file hashes. Offline reproduction requires the complete project folder, including the preserved data, weights and accepted indices. Source-only Git excludes binary artifacts.

## 12. Verified primary references

'''+references+'\n'
    write(REPORT/'FINAL_COMPARATIVE_REPORT.md',header+body+appendix)
    shutil.copytree(PRIMARY/'figures',REPORT/'figures',dirs_exist_ok=True)
    (REPORT/'tables').mkdir(exist_ok=True)
    for name in ['metrics.csv','epoch_history.csv','paired_differences.csv','comparison_summary.csv','hardware_summary.csv']:
        shutil.copy2(PRIMARY/name,REPORT/'tables'/name)
    for path in (RESULTS/'tables').glob('*.csv'):
        shutil.copy2(path,REPORT/'tables'/path.name)
    explained=read(PRIMARY/'results_explained.json')
    explained['final_acceptance']={'description':'Fresh primary results, independent statistical acceptance and separate optimization review. Technical completion is not supervisor approval.','statistics':read(RESULTS/'statistics/phase_5_acceptance.json'),'optimization':read(RESULTS/'statistics/phase_7_acceptance.json'),'project_details':fields,'verified_sources':records,'report':'07_report/FINAL_COMPARATIVE_REPORT.md','technical_phases_completed':list(range(9))}
    save(REPORT/'final_results_explained.json',explained)
    report=(REPORT/'FINAL_COMPARATIVE_REPORT.md').read_text(encoding='utf-8')
    images=re.findall(r'!\[[^]]*\]\(([^)]+)\)',report)
    for image in images:
        require((REPORT/image).is_file(),f'Broken final figure: {image}')
    for token in ['## Abstract']+[f'## {i}.' for i in range(1,13)]:
        require(token in report,f'Missing report section: {token}')
    stats=read(PROJECT_ROOT/'05_experiments/retrained_v1/artifacts/statistical_summary.json')
    for row in stats['summary']:
        require(f"{row['auroc_mean']:.4f} ± {row['auroc_sd']:.4f}" in report,'Report AUROC drift')
        require(f"{row['f1_mean']:.4f}" in report,'Report F1 drift')
    record={'status':'PASS','description':'Core comparison complete with checked local values, verified citations, team details, source links, limitations and seven result figures. This is technical acceptance, not a supervisor sign-off.','report':'07_report/FINAL_COMPARATIVE_REPORT.md','primary_session':'retrained_v1','source_report_sha256':sha(PRIMARY/'comparative_analysis_report.md'),'report_sha256':sha(REPORT/'FINAL_COMPARATIVE_REPORT.md'),'verified_primary_references':len(records),'figure_links_verified':len(images),'project_details_source':'01_planning/PROJECT_DETAILS.md'}
    save(REPORT/'phase_6_acceptance.json',record)
    write(REPORT/'PHASE_6_REPORT_REVIEW.md','# Phase 6 — comparative analysis complete\n\n'+record['description']+'\n\nResNet leads mean AUROC; MobileNet leads F1 at 0.5 and has fewer parameters. The declared architecture differences remain inconclusive. Fresh-session resource tables are separate from the pilot. Supporting tables and figures accompany the report, and the final JSON includes descriptions beside its measurements. No college rubric or signed approval is asserted.\n')
    print('PHASE 6 PASS: final comparative report',flush=True)


def update_tracker():
    path=PROJECT_ROOT/'MILESTONE_TRACKER.md'; text=path.read_text(encoding='utf-8')
    text=re.sub(r'\*\*Current status:\*\*[^\n]*','**Current status:** Phases 0–8 are technically complete. The final report, independent statistics review, optimization note, runnable code and integrity-checked evidence package are ready for supervisor review. College formatting and submission remain handoff tasks.',text)
    evidence={5:('894 numerical checks and accepted outputs','06_results/PHASE_5_STATISTICAL_REVIEW.md'),6:('Report, checked citations, team details and figures','07_report/FINAL_COMPARATIVE_REPORT.md'),7:('Paired learning and disclosed hardware probes','06_results/PHASE_7_OPTIMIZATION.md'),8:('Evidence index, manifest, commands and next experiment','08_handoff/START_HERE.md')}
    lines=[]
    for line in text.splitlines():
        for phase,(note,location) in evidence.items():
            if line.startswith(f'| {phase} |'):
                outcome=line.split('|')[2].strip()
                line=f'| {phase} | {outcome} | COMPLETE (technical) | {note} | `{location}` | {datetime.now().strftime("%d %B %Y")} |'
        if line.startswith('- [ ]'):
            line=line.replace('- [ ]','- [x]',1)
        lines.append(line)
    text='\n'.join(lines)+'\n'
    text=re.sub(r'## Next action\n[\s\S]*$','## Next action\n\nRead `08_handoff/START_HERE.md`, review team details, demonstrate the code, apply the college template if required and present the completed comparison. The proposed batch-32 validation pilot is not executed.\n',text)
    write(path,text)

def package():
    HANDOFF.mkdir(exist_ok=True)
    index=[
        ('Final report','07_report/FINAL_COMPARATIVE_REPORT.md','Fresh-session methods, comparison, costs, uncertainty, limitations and references'),
        ('Explained results','07_report/final_results_explained.json','Measurements and descriptions together, plus accepted review results'),
        ('Training code','04_implementation/src/experiment.py','Models, preprocessing, trainable stages, loss, training, prediction and timing'),
        ('Python settings','04_implementation/src/settings.py','Models, seeds and hyperparameters in inspectable Python'),
        ('Analysis','04_implementation/src/analyze.py','Bootstrap, model scores, original plots and report'),
        ('Independent review','04_implementation/src/review_statistics.py','Weighted-pair bootstrap and raw measurement checks'),
        ('Finalizer','04_implementation/src/finalize_project.py','Acceptance and packaging code; no model retraining'),
        ('Run commands','run.ps1','Full and staged training or saved-evidence reproduction'),
        ('Final review commands','finalize.ps1','Independent statistics, report assembly and integrity verification'),
        ('Run instructions','04_implementation/README.md','Environment installation, data, weights and commands'),
        ('Pipeline methods','04_implementation/PIPELINE.md','Readable experimental flow'),
        ('Reproduction proof','04_implementation/REPRODUCTION_VALIDATION.md','Exact weights and probabilities in the tested environment'),
        ('Statistical review','06_results/PHASE_5_STATISTICAL_REVIEW.md','Uncertainty, verification and throughput limitations'),
        ('Numeric checks','06_results/statistics/phase_5_checks.csv','One row per check with discrepancy and tolerance'),
        ('Accepted comparison','06_results/tables/accepted_comparison.csv','Three core architectures plus a frozen control'),
        ('Precision comparison','06_results/tables/precision_consistency.csv','Identical-real-input probability and decision changes'),
        ('Hardware sessions','06_results/tables/hardware_sessions.csv','Two separate measured sessions'),
        ('Optimization proof','06_results/PHASE_7_OPTIMIZATION.md','Measured learning, memory, timing and capacity evidence'),
        ('Next experiment','05_experiments/NEXT_OPTIMIZATION_EXPERIMENT.md','Validation-only batch-32 plan with fixed criteria; not executed'),
        ('Canonical dataset','archive/pilot_2026-10-22/data/pneumoniamnist_224.npz','Actual 224-pixel data, checksum verified'),
        ('Clean membership','03_data/manifests/train_indices.npy','Accepted training indices; validation/test indices adjacent'),
        ('Fresh provenance','05_experiments/retrained_v1/provenance.json','Launch and final-analysis source hashes'),
        ('Fresh execution log','05_experiments/retrained_v1/execution.log','Actual full-run console record'),
        ('Dependencies','04_implementation/requirements-lock.txt','Exact measured versions; preserved offline dependencies also retained'),
        ('Citation review','02_literature/VERIFIED_SOURCES.md','Verified primary-source scope and relevance'),
        ('Project details','01_planning/PROJECT_DETAILS.md','User-supplied team and supervisor details')]
    csv_out(HANDOFF/'evidence_index.csv',[{'item':a,'relative_path':b,'description':c} for a,b,c in index])
    write(HANDOFF/'EVIDENCE_INDEX.md','# Evidence index\n\n'+table(['Item','Location','Purpose'],[[a,f'[{b}](../{b})',c] for a,b,c in index])+'\n\nComplete checkpoints, NPZ predictions, bootstrap draws and raw timing samples are under `05_experiments/retrained_v1/artifacts/`. The pilot remains identifiable. The manifest includes the preserved dependencies and dataset; source-only Git excludes binary artifacts.\n')
    write(HANDOFF/'START_HERE.md', '''# Completed research package

Technical Phases 0–8 are complete. Start with [the final report](../07_report/FINAL_COMPARATIVE_REPORT.md), [the evidence index](EVIDENCE_INDEX.md) and [the supervisor guide](../07_report/SUPERVISOR_GUIDE.md).

The comparison uses three pretrained CNNs, three seeds and a five-epoch recipe on the accepted 224-pixel pediatric benchmark. Paired ResNet fine-tuning and measured hardware probes provide the optimization proof. Data, pretrained weights, trained checkpoints, predictions, source, dependencies and logs remain in this root.

From this folder:

```powershell
.\run.ps1 verify
.\run.ps1 reproduce-results -RunDir '05_experiments/supervisor_demo_new'
.\finalize.ps1 verify-package
```

Use a fresh run-directory name for new adoption or preparation. `run.ps1 full` performs another training run if needed. `finalize.ps1 all` rebuilds the independent statistical review and final package using CPU evidence checks, without training models.

Preserve the whole root folder for offline handover, including the dated archive and ignored binary artifacts. A source-only checkout cannot supply the trained evidence. The shared CUDA environment can be recreated locally with the documented instructions. No cloud upload or submission was performed.

What you should do: check the supplied names and roll-number mapping, read the limitations, apply a required college template, demonstrate the code and checkpoint replay, and submit through your normal college process. Supervisor approval was not invented. The proposed batch-32 validation pilot remains future work.

Present short hardware timing estimates as engineering probes. Do not claim clinical deployment, universal superiority, equivalence, compression, an absolute GPU batch maximum or guaranteed training acceleration.
''')
    save(HANDOFF/'phase_8_acceptance.json',{'status':'PASS','protocol_id':PROTOCOL_ID,'description':'Technical package assembled with source, final report, data/artifacts, references, environment instructions, explicit next experiment and retained-file hashes. College formatting and submission remain manual tasks.','primary_session':'retrained_v1','next_experiment_executed':False,'new_training_this_phase':False})
    write(HANDOFF/'PHASE_8_HANDOFF.md','# Phase 8 — technical package ready\n\nThe final report, evidence index, environment/reproduction instructions and integrity manifest are present. Pilot and fresh artifacts remain identifiable. The next validation-only optimization has concrete criteria. This technical handoff does not assert college submission or a supervisor signature.\n')
    update_tracker()
    readme=PROJECT_ROOT/'README.md'; current=readme.read_text(encoding='utf-8')
    marker='**Final technical package:** [completed report](07_report/FINAL_COMPARATIVE_REPORT.md), [handoff](08_handoff/START_HERE.md) and [optimization evidence](06_results/PHASE_7_OPTIMIZATION.md). Phases 0–8 are technically complete.\n\n'
    if '**Final technical package:**' not in current:
        first,rest=current.split('\n',1); write(readme,first+'\n\n'+marker+rest.lstrip())
    write(PROJECT_ROOT/'YOUR_NEXT_STEPS.md','# Your next steps\n\n1. Read `07_report/FINAL_COMPARATIVE_REPORT.md` and `08_handoff/START_HERE.md`.\n2. Confirm the supplied team details and roll-number mapping; apply your college template if required.\n3. Demonstrate `run.ps1 verify` and a fresh `reproduce-results` run.\n4. Explain AUROC versus threshold F1, inconclusive architecture intervals and the paired fine-tuning signal.\n5. Present memory savings and session timing variability honestly.\n6. Run `finalize.ps1 verify-package` before handing over the complete folder.\n7. Submit through your college process. The batch-32 pilot is optional future work.\n')
    frozen=read(PROJECT_ROOT/'05_experiments/configs/protocol_freeze_v1.json')
    for item in frozen['files']:
        require(sha(PROJECT_ROOT/item['relative_path'])==item['sha256'],'Frozen evidence changed')
    exclusions={'08_handoff/package_manifest.csv','08_handoff/package_verification.json','08_handoff/package_verification.md','08_handoff/finalize.log'}
    entries=[]
    for path in sorted(PROJECT_ROOT.rglob('*')):
        if not path.is_file(): continue
        relative=path.relative_to(PROJECT_ROOT)
        if any(p in {'.git','.venv','__pycache__'} for p in relative.parts) or path.suffix in {'.pyc','.tmp'}: continue
        if relative.parts[0]=='.cache' or relative.as_posix() in exclusions: continue
        require(path.resolve().is_relative_to(PROJECT_ROOT),'Manifest path escapes root')
        entries.append({'relative_path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path),'role':'preserved_pilot_and_offline_dependencies' if relative.parts[0]=='archive' else 'active_evidence'})
    csv_out(HANDOFF/'package_manifest.csv',entries)
    print(f'PHASE 8 PACKAGE: {len(entries)} retained files inventoried',flush=True)
    return verify_package()


def verify_package():
    path=HANDOFF/'package_manifest.csv'
    require(path.exists(),'Run all to create the package first')
    with path.open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    failures=[]
    for row in rows:
        target=(PROJECT_ROOT/row['relative_path']).resolve()
        require(target.is_relative_to(PROJECT_ROOT),'Unsafe manifest entry')
        if not target.is_file() or target.stat().st_size!=int(row['bytes']) or sha(target)!=row['sha256']:
            failures.append(row['relative_path'])
    result={'status':'FAIL' if failures else 'PASS','files_checked':len(rows),'manifest_sha256':sha(path),'failures':failures,'description':'Listed retained files verified by byte count and SHA-256. Active cache, local virtual environment, Git metadata, Python caches and manifest/verification outputs are excluded. New extra files are not implicitly authenticated by this snapshot.'}
    save(HANDOFF/'package_verification.json',result)
    write(HANDOFF/'package_verification.md',f'# Package integrity\n\n{result["status"]}: {len(rows)} files checked.\n\n{result["description"]}\n\nManifest SHA-256: `{result["manifest_sha256"]}`.\n')
    require(not failures,'Integrity failures: '+', '.join(failures[:8]))
    print(f'PACKAGE PASS: {len(rows)} files match',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['all','review','verify-package'],nargs='?',default='all')
    args=parser.parse_args()
    if args.command=='verify-package':
        verify_package(); return
    from validate_reproduction import validate
    validate()
    stats,hardware,precision=review()
    if args.command=='review': return
    records,references=literature()
    figures(hardware)
    optimization(hardware,precision)
    final_report(records,references)
    package()
    print('PHASES 5-8 TECHNICALLY COMPLETE; no new training',flush=True)


if __name__=='__main__':
    main()