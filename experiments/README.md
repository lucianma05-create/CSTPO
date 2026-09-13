# Cog-Sim / CSTPO diagnostics

Run from the CSTPO workspace. These scripts import the unchanged source in `Cog-Sim/`.
They require its dependencies (including `openai`); no extra statistics package is needed.

```bash
python experiments/audit_cogsim.py
python experiments/branch_probe.py --self-check
python -m py_compile experiments/audit_cogsim.py experiments/branch_probe.py
```

The offline audit writes `results/offline_audit.json`. Its reported FAIL cases are
reproductions, not a failing test runner exit status. See each case's expectation
and observation; integration requirements and audit coverage are separate from
implementation defects.

To run the model pilot, use an existing credential file or set `DEEPSEEK_API_KEY`:

```bash
python experiments/branch_probe.py --env-file /path/to/.env --out experiments/results/new_probe
python experiments/branch_probe.py --summarize experiments/results/new_probe
```

The output directory must not already exist, so old runs cannot be overwritten.
Defaults are three synthetic tasks, 36 single-turn branches, `deepseek-flash`,
three workers, and at most 300 HTTP requests. Each branch is saved immediately.
Do not treat interrupted or incomplete groups as balanced data. To inspect an
interrupted run, the summarize command requires only its manifest and saved branches.
The `calls` field is the original client's successful usage counter; `http_requests`
also counts compatibility fallback attempts, with SDK retries disabled. Reported
tokens are provider-returned usage and cannot account for unreturned failed requests.

The synthetic strategy set and wording choices are diagnostic interventions, not
validated dataset labels. All roots have empty dialogue histories. Sensitivity here
is a descriptive property of that finite stimulus set at that root, not an estimate
over a learned policy. New state nodes remain in the raw logs but are excluded from
parent-anchor distances. Full cognitive-state sensitivity, multi-turn checkpoints,
human fidelity, terminal return prediction, and downstream RL remain untested.

Protocol and interpretation: [CogSim审计与CSTPO先导实验](../shared_work_space/90_CogSim历史审计与先导记录.md).
