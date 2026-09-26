# Proactive Dialogue Policy Optimization via Cognitive-State Transition

**ICLR 2027 submission — OpenReview #39751** (anonymized repository)

CSTPO is a proactive dialogue policy optimization framework that couples a
cognitive-state-driven user simulator (Cog-Sim) with a two-level policy trained by
SFT + PPO. The policy emits a strategy label and an utterance at every turn; a
dual-head Cog-critic estimates separate values for the strategy field and the
utterance field, and a tree-branching rollout budget (trunk + node continuations)
allocates exploration to unresolved dialogue states.

Tasks: **ESConv** (emotional support), **PersuasionForGood / P4G** (charity
persuasion), and **CraigslistBargain** (price negotiation).

## Repository Structure

```text
cstpo/
  core/       task env, agent, judge, label maps, seed adapter, terminal rewards
  rl/         verl agent loop (two-field generation + tree branching), reward bridge
  sft/        SFT data building, training, tokenization, label trie samplers
  seedgen/    seed extraction/annotation/review pipeline (LLM-assisted)
  eval/       judge/diversity/sensitivity probes and calibration tools
  p0/         phase-A field-advantage and cost analyses
  configs/    agent loop registration (cstpo_agent_loop.yaml), RL hyperparameters
scripts/      entry scripts (RL launch, SFT snapshot, data building, eval)
docs/         verl 0.8 patch inventory (VERL_PATCHES.md)
Cog-Sim/      cognitive user simulator (BDI + emotion transition engine)
result/       experiment documents (local-only, not distributed)
data/         seeds and evaluation sets (local-only, not distributed)
```

## Quick Start

### 1. Environment

```bash
conda create -n verl python=3.12 -y
conda activate verl
pip install verl==0.8 vllm torch
pip install -e .
```

Source the environment exports before launching any verl job:

```bash
source scripts/verl_env.sh
```

### 2. API keys

Copy the placeholder env file and fill in your DeepSeek API key (used by the judge,
the user simulator, and the annotation pipeline):

```bash
cp Cog-Sim/.env.example Cog-Sim/.env
# edit Cog-Sim/.env: DEEPSEEK_API_KEY=sk-...
```

`Cog-Sim/.env` is git-ignored and never distributed.

### 3. SFT

Build the SFT training set from seeds (strategy label + utterance two-field format):

```bash
python -m cstpo.sft.build_sft --task esconv     # esconv | p4g | craigslistbargain
python -m cstpo.sft.train_sft --task esconv
```

SFT hyperparameters live in `cstpo/sft/sft_config.py`.

### 4. RL (PPO + dual-head Cog-critic)

```bash
bash scripts/mc_ppo_critic_run.sh
```

Hyperparameters are centrally managed in `cstpo/configs/rl_defaults.sh`
(override via environment variables, e.g. `STEPS=40 bash scripts/mc_ppo_critic_run.sh`).
Key settings: 10 seeds/step, trunk + 4/3 branch nodes × 2 continuations per seed
(esconv 9 / cb-p4g 7 rollouts), 30-turn cap, γ = 1, actor LR 1e-5.
Training requires the verl 0.8 patches listed in `docs/VERL_PATCHES.md`.

### 5. Evaluation

The free100 protocol evaluates 100 frozen cases per task against a Cog-Sim user
simulator with judge rev4 (n=3 aggregation), cluster-bootstrap CIs, and two-level
paired diffs. The baseline runner lives in the local-only `baseline/` directory and
is not distributed with the repository.

## Reproducibility

- `docs/VERL_PATCHES.md` lists every local patch applied to verl 0.8 (dual-head
  value wrapper, two-level GAE, label choice-mask renormalization, critic resource
  pool, health probes). Re-apply these patches before training on a new machine.
- Datasets, seeds, experiment logs, and result documents are kept locally and are
  not distributed with the repository (see `.gitignore`); only pipeline code is
  version-controlled.
- Cog-Sim is vendored as a source copy (not a git submodule); cloning this
  repository includes the simulator source.
