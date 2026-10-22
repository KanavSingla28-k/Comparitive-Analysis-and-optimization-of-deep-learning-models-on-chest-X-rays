# Completed research package

Technical Phases 0–8 are complete. Start with [the final report](../07_report/FINAL_COMPARATIVE_REPORT.md), [the evidence index](EVIDENCE_INDEX.md) and [the supervisor guide](../07_report/SUPERVISOR_GUIDE.md).

The comparison uses three pretrained CNNs, three seeds and a five-epoch recipe on the accepted 224-pixel pediatric benchmark. Paired ResNet fine-tuning and measured hardware probes provide the optimization proof. Data, pretrained weights, trained checkpoints, predictions, source, dependencies and logs remain in this root.

From this folder:

```powershell
.un.ps1 verify
.un.ps1 reproduce-results -RunDir '05_experiments/supervisor_demo_new'
.inalize.ps1 verify-package
```

Use a fresh run-directory name for new adoption or preparation. `run.ps1 full` performs another training run if needed. `finalize.ps1 all` rebuilds the independent statistical review and final package using CPU evidence checks, without training models.

Preserve the whole root folder for offline handover, including the dated archive and ignored binary artifacts. A source-only checkout cannot supply the trained evidence. The shared CUDA environment can be recreated locally with the documented instructions. No cloud upload or submission was performed.

What you should do: check the supplied names and roll-number mapping, read the limitations, apply a required college template, demonstrate the code and checkpoint replay, and submit through your normal college process. Supervisor approval was not invented. The proposed batch-32 validation pilot remains future work.

Present short hardware timing estimates as engineering probes. Do not claim clinical deployment, universal superiority, equivalence, compression, an absolute GPU batch maximum or guaranteed training acceleration.
