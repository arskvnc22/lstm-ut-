# Storing Is Not Remembering: LSTM-UT and Bounded Gated Memory for Looped Transformers

**Aras Kavuncu, Muhammad Burhan Hafez** (University of Southampton)

[arXiv:2609.19521](https://arxiv.org/abs/2609.19521) · [PDF](https://arxiv.org/pdf/2609.19521)

Recurrent-depth (looped) Transformers reuse one block across many steps, so information needed later must survive repeated
rewriting of the hidden state. This repository contains the code for a controlled comparison of three ways of carrying
information across recurrent steps:

- **BUT** (Block Universal Transformer) keeps only its current hidden sequence.
- **CoTFormer** additionally attends to the cached keys and values of every earlier repeat.
- **LSTM-UT** (introduced in the paper) adds a small, bounded, gated cell state to the shared block. The block's own
  proposal is kept intact, and memory is read through an additive output gate.

The models are compared on Rule 30 depth extrapolation, delayed cellular-automaton recall, and OpenWebText2 language
modelling.

## Headline results

| Experiment | Result |
|---|---|
| Rule 30 extrapolation, trained to depth 12 | LSTM-UT: 99.67 ± 0.56% exact-row accuracy at depth 60. BUT, at width 64 or parameter-matched width 80, gets no row fully correct. |
| Long-horizon stress test | One R12 LSTM-UT checkpoint stays above 99.95% exact-row accuracy through depth 1,000 (10 of 61,440 rows wrong). |
| Delayed recall, H_max = 12, two recall applications | Cell accuracy 90.4% (LSTM-UT), 79.5% (BUT), 60.0% (CoTFormer). Width-48 LSTM-UT, with 6.5% fewer parameters than BUT, still reaches 88.9%. |
| CoTFormer cache diagnostics | Hiding the cache block written at the requested step changes recall accuracy by at most 0.05 points. |
| OpenWebText2, width 384 | LSTM-UT beats BUT at R = 5, 8, 10 by 1.2–1.3 perplexity and is slightly below CoTFormer, which needs up to 91% more time per step. Against a parameter-matched CoTFormer, LSTM-UT is within 0.6 perplexity. |

The paper gives the full tables, seeds, and caveats.

## Repository layout

Only the parts used by the paper are listed.

```text
cellular_automaton/          Rule 30 data, training, evaluation and analysis
  ca_main.py                 entry point: Rule 30 depth extrapolation
  dca_main.py                entry point: delayed cellular-automaton recall (DCA)
  ca_analyze.py              run aggregation and tables (python -m cellular_automaton.ca_analyze)
  ca_attention_diagnostics.py   CoTFormer state/cache factorial interventions
  ca_intervention_paper_data.py aggregation of factorial-intervention summaries
  dca_intervene_eval.py      delayed-recall cache interventions
  test_*.py                  unit tests
models/
  lstm_ut_bidir.py           LSTM-UT for Rule 30
  but_full_depth.py          BUT for Rule 30
  ca_cotf.py                 CoTFormer for Rule 30 (SDPA attention)
  ca_cotf_cache_attn.py      CoTFormer with manual attention and a repeat-cache window
  dca_lstm_ut.py, dca_but.py, dca_cotf_cache.py      delayed-recall variants
  but_llm.py, cotformer_llm.py, lstm_ut_all_core.py  OpenWebText2 language models
  ln_cotformer_llm.py, ln_lstm_ut_all_core.py        "LN-" variants (LayerNorm after every repeat)
main.py                      entry point: OpenWebText2 language modelling
config/base.py               shared argument definitions
data/openwebtext2.py         OpenWebText2 extraction and tokenisation
iridis/                      Slurm launchers (University of Southampton Iridis X)
  env.sh                     shared paths (scratch, dataset, caches, conda env)
  ca-rule30/                 Rule 30 launchers: job_but.sh, job_cotformer.sh, job_ca_lstm_ut.sh
  dca-30/                    delayed-recall launchers: job_dca_{but,cotf,lstm_ut}.sh, job_dca_intervene_eval.sh
  owt2-llms/                 language-model launchers
  download-dataset/, extract-tokenize-dataset/   OpenWebText2 preparation
```

## Setup

```bash
conda create -n ml python=3.11 -y
conda activate ml
pip install -r requirements.txt   # PyTorch 2.2.0 (CUDA 11.8), numpy<2, tiktoken, ...
```

All launchers were written for the Iridis X Slurm cluster. Before running them elsewhere:

- Edit `iridis/env.sh`. Its shared scratch root (`/scratch/ab3u21`), conda environment, dataset and cache directories are
  hard-coded.
- Set `PARTITION`, `ACCOUNT`, `MEMORY` and `TIME_LIMIT` as environment variables. The Rule 30 and delayed-recall launchers
  read them.
- Change the `#SBATCH` header of each `iridis/owt2-llms/*.sh` you use.
- Edit the hard-coded `EXPS_DIR` checkpoint directories in `iridis/ca-rule30/job.sh`, `iridis/dca-30/job.sh` and the
  `iridis/owt2-llms/*.sh` scripts.

Every launcher is run with `bash` from the repository root. Outside a Slurm allocation it submits itself with `sbatch`.

## Rule 30 depth extrapolation (paper Section 4)

A model receives a 64-cell row and applies one shared block per Rule 30 update. Under protocol R_K, training cycles
deterministically through depths `1:1 … K:K`, one depth per optimizer step, and only the final row is supervised. The
checkpoint used for reporting (`best_extrapolation_unconstrained`) maximises validation cell accuracy at one unseen depth:
9 for R6, 11 for R8 and 15 for R12. Final results come from a separate 4,096-row test split.

Each run writes `run_manifest.json` (fully resolved configuration) and `eval_metrics.jsonl` (every scalar metric) to
`iridis/ca-rule30/runs/run_N__<name>/`. Checkpoints go to the scratch results directory.

Shared settings for every Rule 30 run in the paper (width 64, four heads, one shared block, `0/1/0` layout):

```bash
COMMON=(
    --n_embd 64 --n_head 4 --n_layer 1 --n_layer_begin 0 --n_layer_end 0
    --batch_size 128 --acc_steps 1 --opt adamw --scheduler none --lr 1e-3
    --grad_clip 0.9 --dropout 0 --dtype torch.bfloat16
    --data_seed 1 --ca_shuffle_seed 1 --ca_val_seed 1000003 --ca_test_seed 2000003
    --ca_val_samples 4096 --ca_test_samples 4096 --ca_eval_max_batches 32
)

R6=(
    --ca_train_pairs 1:1 2:2 3:3 4:4 5:5 6:6 --ca_best_pair 6:6
    --ca_extrapolation_val_pairs 9:9 --ca_extrapolation_best_pair 9:9
    --iterations 7500 --eval_freq 500 --weight_decay 0.10 --ca_train_samples 100000
)

R8=(
    --ca_train_pairs 1:1 2:2 3:3 4:4 5:5 6:6 7:7 8:8 --ca_best_pair 8:8
    --ca_extrapolation_val_pairs 9:9 10:10 11:11 12:12 13:13 14:14 15:15 16:16
    --ca_extrapolation_best_pair 11:11
    --iterations 10000 --eval_freq 250 --weight_decay 0.10 --ca_train_samples 100000
)

R12=(
    --ca_train_pairs 1:1 2:2 3:3 4:4 5:5 6:6 7:7 8:8 9:9 10:10 11:11 12:12 --ca_best_pair 12:12
    --ca_extrapolation_val_pairs 13:13 14:14 15:15 16:16 17:17 18:18 19:19 20:20
    --ca_extrapolation_best_pair 15:15
    --iterations 24000 --eval_freq 500 --weight_decay 0.10 --ca_train_samples 500000
)

LSTM=(--lstm_forget_gate learned --lstm_control_input previous_and_proposed --lstm_initial_cell zero)
```

The paper uses five model seeds (`--seed 0` … `--seed 4`). Trailing arguments override earlier ones, so `--n_embd`
can be appended to change the width.

> **Important:** `iridis/ca-rule30/job.sh` defaults `REPEAT_CACHE_WINDOW` to 4, and that setting is only valid for
> `ca_cotf_cache_attn`. Pass `REPEAT_CACHE_WINDOW=` (empty) for every BUT and CoTFormer run.
> `job_ca_lstm_ut.sh` already does this.

### R6 (paper Figure 1 left, Table 7)

```bash
for SEED in 0 1 2 3 4; do
  REPEAT_CACHE_WINDOW= CA_FINAL_EVAL_PAIRS= DIAGNOSTIC_MAX_REPEATS=12 TIME_LIMIT=03:00:00 \
    bash iridis/ca-rule30/job_ca_lstm_ut.sh "${COMMON[@]}" "${R6[@]}" "${LSTM[@]}" \
      --ca_final_repeat_diagnostic_max_repeats 40 --seed $SEED
  REPEAT_CACHE_WINDOW= CA_FINAL_EVAL_PAIRS= DIAGNOSTIC_MAX_REPEATS=12 TIME_LIMIT=03:00:00 \
    bash iridis/ca-rule30/job_but.sh "${COMMON[@]}" "${R6[@]}" \
      --ca_final_repeat_diagnostic_max_repeats 25 --seed $SEED
  REPEAT_CACHE_WINDOW= CA_FINAL_EVAL_PAIRS= DIAGNOSTIC_MAX_REPEATS=12 TIME_LIMIT=03:00:00 \
    bash iridis/ca-rule30/job_cotformer.sh "${COMMON[@]}" "${R6[@]}" \
      --ca_final_repeat_diagnostic_max_repeats 20 --seed $SEED
done
```

For the width-32 LSTM-UT row, append `--n_embd 32` to the LSTM-UT command.

### R8 and R12 (paper Figure 1 right, Table 8)

```bash
for SEED in 0 1 2 3 4; do
  # R12: use "${R12[@]}" and DIAGNOSTIC_MAX_REPEATS=20.
  # R8:  use "${R8[@]}"  and DIAGNOSTIC_MAX_REPEATS=16.
  REPEAT_CACHE_WINDOW= CA_FINAL_EVAL_PAIRS= DIAGNOSTIC_MAX_REPEATS=20 TIME_LIMIT=03:00:00 \
    bash iridis/ca-rule30/job_ca_lstm_ut.sh "${COMMON[@]}" "${R12[@]}" "${LSTM[@]}" \
      --ca_final_repeat_diagnostic_max_repeats 60 --seed $SEED
  REPEAT_CACHE_WINDOW= CA_FINAL_EVAL_PAIRS= DIAGNOSTIC_MAX_REPEATS=20 TIME_LIMIT=03:00:00 \
    bash iridis/ca-rule30/job_but.sh "${COMMON[@]}" "${R12[@]}" \
      --ca_final_repeat_diagnostic_max_repeats 60 --seed $SEED
done
```

Parameter-adjusted rows: append `--n_embd 80` to the BUT command (77,200 parameters) and `--n_embd 48` to the LSTM-UT
command (46,416 parameters).

### CoTFormer cache windows and interventions (paper Section 4.3, Tables 1 and 10)

Bounded cache visibility during training uses the manual-attention CoTFormer under the R12 protocol. `k` ranges over
1–8, and an empty `REPEAT_CACHE_WINDOW` gives the full cache:

```bash
MODEL=ca_cotf_cache_attn ATTENTION_IMPLEMENTATION=manual REPEAT_CACHE_WINDOW=$k \
CA_FINAL_EVAL_PAIRS= TIME_LIMIT=03:00:00 \
  bash iridis/ca-rule30/job_cotformer.sh "${COMMON[@]}" "${R12[@]}" --seed $SEED
```

The clean-state/cache-reset factorial intervention runs on one frozen checkpoint at a chosen source depth. It has four
paired conditions: free baseline, clean state, cache reset, and both. It is implemented in
`python -m cellular_automaton.ca_attention_diagnostics`; see `--help`. The main arguments are `--run-manifest`,
`--checkpoint-name`, `--expected-step`, `--intervention-source-depth` and `--evaluate-factorial-full-split`.
`python -m cellular_automaton.ca_intervention_paper_data` turns the resulting `summary.json` files into the paired effects
reported in Table 1.

## Delayed cellular-automaton recall (paper Section 5)

The model runs H Rule 30 updates, then is queried for an earlier row X^(q) through a learned direction/age controller. One
or two additional shared-block applications produce the answer. Horizons `1:1 … H_max:H_max` cycle deterministically, and
the loss is `(L_evo + 2 L_query) / 3`. The checkpoint (`best_delayed_recall`) is chosen among steps 0, 2,500 and 5,000
by pair-balanced validation recall cell accuracy.

The main sweep crosses H_max ∈ {6, 12, 20}, recall applications ∈ {1, 2}, and model seeds {1, 2, 3}, with data seed 1
shared throughout:

```bash
H=12; RECALL=2; SEED=1
PAIRS=(); for h in $(seq 1 $H); do PAIRS+=("$h:$h"); done
DCA=(
    --n_embd 64 --n_layer 1 --iterations 5000 --eval_freq 2500
    --ca_train_pairs "${PAIRS[@]}" --ca_best_pair "$H:$H" --ca_max_relative_age $H
    --ca_delayed_percentage 100 --query_horizon_policy uniform --ca_query_loss_weight 2
    --ca_controller_application subtract --ca_recall_repeats $RECALL
    --seed $SEED --data_seed 1 --ca_shuffle_seed 1 --use_pretrained None
)

bash iridis/dca-30/job_dca_but.sh     "${DCA[@]}"
bash iridis/dca-30/job_dca_cotf.sh    "${DCA[@]}"
bash iridis/dca-30/job_dca_lstm_ut.sh "${DCA[@]}" \
    --lstm_forget_gate learned --lstm_control_input previous_and_proposed
```

The delayed-recall presets fix the remaining settings: four heads, batch size 128, AdamW at learning rate 1e-3, weight
decay 0.15, gradient clipping at 0.9, no schedule, 100,000 training rows, and 4,096 validation and 4,096 test rows. Runs
are written to `iridis/dca-30/runs/`.

**Parameter-matched comparison (Section 5.4, Table 4):** run the LSTM-UT command with `--n_embd 48` and compare it with
width-64 BUT. That sweep used `--eval_freq 1000`.

**CoTFormer cache interventions (Section 5.3, Table 3):** permuting or hiding the requested cache block, or leaving it as
the only visible evolution block, is run on a trained delayed-recall CoTFormer checkpoint:

```bash
SOURCE_RUN_ID=<run_N__dca_cotf_...> EXPECTED_STEP=5000 HORIZONS="20" \
  bash iridis/dca-30/job_dca_intervene_eval.sh
```

## Language modelling on OpenWebText2 (paper Section 6)

### Data

```bash
bash iridis/download-dataset/job.sh             # login node: downloads the OWT2 mirror and the tiktoken BPE
bash iridis/extract-tokenize-dataset/job.sh     # compute node: writes train.bin / val.bin under $DATA_DIR/openwebtext2
```

### Training

Every model has a 2-layer prelude, a 9-layer looped core and a 1-layer coda, with six heads and sequence length 256.
Training runs 40,000 AdamW steps at learning rate 1e-3 with 20% warmup and cosine decay, and an effective batch size of
256 on two GPUs. Each configuration uses one seed (`--seed 0`). The jobs auto-resume from the latest checkpoint and write
logs to `iridis/owt2-llms/run_N/`.

| Paper row | Launcher | `--model` |
|---|---|---|
| BUT | `iridis/owt2-llms/but_job.sh` | `but_llm` |
| CoTFormer | `iridis/owt2-llms/cotf_job.sh` | `cotformer_llm` |
| LSTM-UT | `iridis/owt2-llms/lstm_all_core_job.sh` | `lstm_ut_all_core` |
| LN-CoTFormer | `iridis/owt2-llms/nat_ln_cotf_job.sh` | `ln_cotformer_llm` |
| LN-LSTM-UT | `iridis/owt2-llms/nat_ln_lstm_all_core_job.sh` | `ln_lstm_ut_all_core` |

The `natapat_*_job.sh` scripts are the same BUT, CoTFormer and LSTM-UT configurations set up for H200 nodes.

The repeat count and width are set at the top of each script. Set `N_REPEAT` to 5, 8 or 10 and `N_EMBD` to 384. For the
parameter-matched R = 5 comparison, set `N_EMBD=408` for BUT and CoTFormer (44.51M parameters, against 45.28M for width-384
LSTM-UT). Then submit:

```bash
bash iridis/owt2-llms/lstm_all_core_job.sh
```

Validation perplexity is logged every 2,000 steps. The paper reports the value at step 40,000 and the median time per
step after step 1,000.

## Analysing results

`ca_analyze` reads run manifests recursively. It only aggregates runs whose non-averaged configuration fields match.
For example, here is the R12 LSTM-UT final-test exact-row accuracy averaged over seeds:

```bash
python -m cellular_automaton.ca_analyze table \
  --runs iridis/ca-rule30/runs \
  --where model=lstm_ut_bidir --where status=completed \
  --where 'training_pairs=1:1|2:2|3:3|4:4|5:5|6:6|7:7|8:8|9:9|10:10|11:11|12:12' \
  --split final_test --role repeat_horizon_diagnostic \
  --checkpoint best_extrapolation_unconstrained \
  --metric exact_sequence_accuracy \
  --report-pairs 12:12 20:20 40:40 60:60 \
  --average-over seed
```

For delayed recall, `delayed-compare` and `recall-repeat-compare` report ground-truth, internal and selection
diagnostics. Run `python -m cellular_automaton.ca_analyze --help` for every subcommand, and see
`iridis/ca-rule30/README.md` for the run-record format.

## Tests

```bash
python -m pytest cellular_automaton
```

The tests check software behaviour, such as data generation, evaluation bookkeeping and analysis. They do not validate
the scientific claims.

## Citation

```bibtex
@article{kavuncu2026storing,
  title   = {Storing Is Not Remembering: {LSTM-UT} and Bounded Gated Memory for Looped Transformers},
  author  = {Kavuncu, Aras and Hafez, Muhammad Burhan},
  journal = {arXiv preprint arXiv:2609.19521},
  year    = {2026}
}
```

## Acknowledgements and license

This code builds on the [CoTFormer](https://github.com/epfml/CoTFormer) codebase (EPFL Machine Learning and Optimization
Laboratory), and its MIT license is kept in `LICENSE`. We thank the University of Southampton for access to the Iridis X
high-performance computing cluster.
