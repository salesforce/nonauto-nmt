# Non-Autoregressive Transformer
Code release for [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281) by Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher.

Requires PyTorch 0.3, torchtext 0.2.1, and SpaCy.

The pipeline for training a NAT model for a given language pair includes:
1. `run_alignment_wmt_LANG.sh` (runs `fast_align` for alignment supervision)
2. `run_LANG.sh` (trains an autoregressive model)
3. `run_LANG_decode.sh` (produces the distillation corpus for training the NAT)
4. `run_LANG_fast.sh` (trains the NAT model)
5. `run_LANG_fine.sh` (fine-tunes the NAT model)
