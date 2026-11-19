# HAESR

PyTorch implementation of HAESR for lightweight single image super-resolution.

## Environments

- Python 3.10.16
- PyTorch 2.6.0

**Dependencies:**

- torchvision
- opencv-python
- numpy
- Pillow
- PyYAML
- tqdm
- einops
- tensorboardX
- thop

Run all commands from the repository root.

## Training

Prepare [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and the validation datasets (Set5, Set14, B100, and Urban100), then set their paths in [env/env.json](env/env.json). Training settings are provided in `train_yamls/`.

```bash
# ×2
python train.py -v HAESR_X2_DIV2K -p train -c 0 --train_yaml train_HAESR_X2_DIV2K.yaml

# ×3
python train.py -v HAESR_X3_DIV2K -p train -c 0 --train_yaml train_HAESR_X3_DIV2K.yaml

# ×4
python train.py -v HAESR_X4_DIV2K -p train -c 0 --train_yaml train_HAESR_X4_DIV2K.yaml
```

`-c` selects the GPU index. Checkpoints and experiment configurations are saved under `train_logs/<experiment_name>/` by default.

## Testing

Set the benchmark paths in [env/env.json](env/env.json). The test script requires a trained checkpoint and its matching `model_config.json`, generated during training:

```text
train_logs/HAESR_X4_DIV2K/
├── model_config.json
└── checkpoints/
    └── epoch1_HAESR.pth
```

Evaluate a checkpoint, for example:

```bash
python test.py -v HAESR_X4_DIV2K -t tester_Matlab -s 1 -c 0 --test_dataset_name Set5
```

Use the corresponding experiment name for ×2 or ×3, and replace `1` with the available checkpoint epoch.

## Acknowledgements

We thank the authors of [ASID](https://github.com/saturnian77/ASID) and [Omni-SR](https://github.com/Francis0625/Omni-SR) for making their research and code publicly available.

## Citation

If you find this work useful, please cite:

```bibtex
@article{LI2027115364,
  title = {Multi-pattern hybrid-attention network for lightweight image super-resolution},
  author = {Li, Mingyang and Liu, Yixiao and Qing, Linbo and He, Xiaohai and Ren, Chao},
  journal = {Pattern Recognition},
  volume = {185},
  pages = {115364},
  year = {2027},
}
```
