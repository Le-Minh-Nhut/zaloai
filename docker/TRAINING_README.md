# LifeMate training reproduction

Baseline: MiniCPM5-2B, SFT LoRA, BF16; see `configs/training_lora.yaml`.
Training examples are generated from the official LifeMate train split using
`scripts/prepare_dataset.py`. The training split and the base model must be
acquired separately from their official sources; record immutable URLs and hashes
for the actual released training dataset and weights before submission. Training
and inference use the code and configs included here. The selected adapter,
checkpoint identity and exact commands must be recorded after training finishes.
See the repository documentation for the dataset preparation and resume commands.
