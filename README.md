# AMP-HGNAS

This repository contains a PyTorch implementation of the preprint paper:  [Adaptive Meta-Path-based Neural Network Architecture Search for
Heterogeneous Graphs](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5502763).

## Requirements

Python 3.9.18 is recommended. Install dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..

## Data preparation

For HGB datasets:

```
sh download_hgb_datasets.sh
```

For experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from OGB challenge.

## Run AMP-HGNAS

You can run AMP-HGNAS on HGB and ogbn-mag based on the command in [hgb](https://github.com/lxin23751-bit/AMP-HGNAS/tree/main/hgb) and [ogbn-mag](https://github.com/lxin23751-bit/AMP-HGNAS/tree/main/ogbn), respectively.

## Cite

If you use AMP-HGNAS in a scientific publication, we would appreciate citations to the following paper:

```
@article{li2026adaptive,
  title  = {Adaptive Meta-Path-Based Neural Network Architecture Search for Heterogeneous Graphs},
  author  = {X. Li and P. Li{\'o} and L. Yang and Z. Ye and C. Peng},
  journal= {Information Sciences},
  year   = {2026},
  pages  = {123554}
}
```

markdown
The structure and some instructions of this README are adapted from the [LMSPS](https://github.com/JHL-HUST/LMSPS) repository.
