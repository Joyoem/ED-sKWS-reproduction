# ED-sKWS Reproduction

Independent reproduction and modern PyTorch port of **ED-sKWS: Early-Decision Spiking Neural Networks for Rapid and Energy-Efficient Keyword Spotting** (Interspeech 2024).

This repository is based on the original ED-sKWS implementation by Song et al. and is **not the official repository**.

Original implementation:  
https://github.com/Pooookeman/ED-sKWS

Paper:  
Z. Song et al., *ED-sKWS: Early-Decision Spiking Neural Networks for Rapid and Energy-Efficient Keyword Spotting*, Interspeech 2024.

## Reproduction Result

We reproduce the 512-hidden-unit ED-sKWS configuration on Google Speech Commands v2 with 35 keyword classes.

| Metric | Paper | This reproduction |
|---|---:|---:|
| Dataset | GSC v2, 35 classes | GSC v2, 35 classes |
| Temporal steps | 98 | 98 |
| Hidden size | 512 | 512 |
| Training epochs | 100 | 100 |
| Late accuracy | 93.15% | **92.56%** |
| Accuracy gap | — | **−0.59 pp** |
| Reported decision timestep | 60.46 | approximately 60–61 |
| Training hardware | — | NVIDIA RTX 3060 |
| Training time | — | approximately 5 h 43 min |

The reproduced late accuracy is within 0.59 percentage points of the result reported in the paper.

## Environment

The original repository targets an older PyTorch stack. This reproduction was run using a modern environment:

- Python 3.9
- PyTorch 2.8.0 + CUDA 12.8
- TorchAudio 2.8.0
- NumPy 1.22.3
- pandas 1.5.3
- h5py 3.7.0

See `requirements-modern.txt` for the complete environment.

## Compatibility Changes

Only compatibility and execution-related changes were introduced. The ED-sKWS architecture, training loss, optimization protocol, and GSC 35-class task definition were not intentionally modified.

Main changes include:

- Disabled an unused SC-100 dataset import to avoid unnecessary dependencies.
- Added compatibility for loading full-module checkpoints under modern PyTorch.
- Dropped the incomplete final training batch to avoid a fixed-batch-size mismatch in the original implementation.
- Added Slurm training scripts and reproducibility metadata.

The exact code changes relative to the original implementation are stored in:

`reproduction_notes/modern_port.patch`

## Evaluation Note

The original evaluation code accumulates some early-decision statistics over batches and divides them by the final zero-based batch index (`step`) rather than the number of batches (`step + 1`).

Therefore, the raw logged early-decision accuracy should not be treated as a corrected sample-level metric.

For the reproduction result reported above, **late accuracy** is taken directly from the final temporal step of `test_acc1.txt`.

A corrected sample-level early-decision evaluator can be added separately.

## Checkpoint

The reproduced checkpoint is stored at:

`checkpoints/gsc35_adlif512_ep100/best_model.pth`

Associated files include:

- `test_acc1.txt`
- `test_acc2.txt`
- `SHA256SUMS`

The checkpoint corresponds to the best validation model obtained during the 100-epoch reproduction run.

## Training

Example Slurm configuration:

`scripts/edskws_gsc35_100.slurm`

The experiment uses:

- adLIF neurons
- 3 layers including the readout layer
- 512 hidden units
- batch size 128
- 100 epochs
- Adam optimizer
- learning rate 0.01
- 40-bin filter-bank input features
- 98 temporal steps

## Next Step

The 35-class experiment is retained as a **paper-fidelity reproduction**.

For comparison with other keyword-spotting architectures such as RepTor, BC-ResNet, and DS-CNN, a separate unified GSC v2 12-class protocol will be used.

This separation avoids conflating reproduction of the original ED-sKWS experiment with our later apples-to-apples benchmark.

## Acknowledgements

This repository builds on the official ED-sKWS implementation and the SpArch framework. Please cite the original ED-sKWS paper and repository when using this reproduction.
