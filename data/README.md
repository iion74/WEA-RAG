# Data directory

Run `bash scripts/prepare_data.sh` to download the public HotpotQA files and
create the JSONL inputs used by the evaluators. Dataset files are intentionally
not committed to this repository.

Expected generated files:

- `hotpot_train_v1.1.json`
- `hotpot_dev_distractor_v1.json`
- `hotpot_dev_fullwiki_v1.json`
- `hotpot_dev_distractor_v1.jsonl`
- `hotpot_dev_fullwiki_v1.jsonl`

HotpotQA is distributed under CC BY-SA 4.0. See
<https://hotpotqa.github.io/> for dataset details and citation information.
