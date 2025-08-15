# DxD_NMT
Determinacy to Defnitness Construal (CG) in PL to EN translation (NMT).   

Outputs: 

| **Model key**  | **mode**      | **pool\_type** | **Sampled / sent.** | **Output / sent.** | **Saved (where)**                                                   |

| --------------- | ------------- | -------------: | ------------------: | -----------------: | --------------------------------------------------------------------|
| Model*          | plain         |              — |                   0 |     1 (top-1 beam) | `selected_summary.parquet`; wide CSV column                         |
| Model*          | beam\_k       |           beam |                 `K` |                `K` | `runs.parquet` (`N×K` rows)                                         |
| Model*          |ext\_mbr       |         sample |                 `S` |      1 (consensus) | `runs.parquet` (`N×S`); `selected_summary.parquet`; wide CSV column |
| Model*          | mbr           |              — |                 `Z` |      1 (zMBR pick) | `selected_summary.parquet`; wide CSV column                         |
| Model*          | mbr\_internal | zmbr\_internal |                 `Z` |                `Z` | `{tag}_zmbr_pools.parquet` (`N×Z`)                                  |
| Google baseline | plain         |              — |                   0 |                  1 | `selected_summary.parquet`; wide CSV column                         |

*Models: marian \| mt5\_seq \| mt5\_cg \| mbart\_seq \| mbart\_cg

| Path                   | `mode`         | `pool_type`                    | What’s saved                                                        |
| ---------------------- | -------------- | ------------------------------ | ------------------------------------------------------------------- |
| Plain beam (top-1)     | `plain`        | —                              | Selected → `translations_all.csv`, `selected_summary.parquet`       |
| Plain beam (top-K)     | `beam_k`       | `beam`                         | All K → `runs.parquet`                                              |
| External MBR (samples) | `ext_mbr`      | `sample` / `sample_plain_twin` | All S → `runs.parquet`; Selected → CSV + `selected_summary.parquet` |
| zMBR selected          | `mbr`          | —                              | Selected → CSV + `selected_summary.parquet`                         |
| zMBR internal pool     | `mbr_internal` | `zmbr_internal`                | All Z → `{tag}_zmbr_pools.parquet`                                  |


| **Artifact**               | **Contents**                                                 |
| -------------------------- | ------------------------------------------------------------ |
| `translations_all.csv`     | Wide per-sentence columns (beam top-1 and selected outputs)  |
| `runs.parquet`             | Long table: all beam-`K` and external-MBR sampled candidates |
| `selected_summary.parquet` | Selected-only rows (google, beam top-1, ext\_mbr, mbr)       |
| `{tag}_zmbr_pools.parquet` | zMBR internal samples (per wrapped model)                    |
