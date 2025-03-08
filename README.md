# `borzoi_loader`

This repo is made to preprocess .bigWig data to be used to finetune Borzoi, and to provide Pytorch DataLoader structures to do so.

This git repo is mostly adapted from my `enformer_loader` repo, but using defaults compatible with `borzoi-pytorch` instead of `enformer-pytorch`.

## Acknowledgements

This repo builds primarily on the work done in the repo enformer-finetune (https://github.com/wconnell/enformer-finetune, License: Apache 2.0, wconnell), and predicts using the borzoi-pytorch implementation of Borzoi (https://github.com/johahi/borzoi-pytorch, License: Apache 2.0, johani). Other functions taken from enformer-pytorch (https://github.com/lucidrains/enformer-pytorch/tree/main, License: MIT, lucidrains).

## Installation

1. Pull the repo, `cd` into repo dir
2. `pip install -e .`

## Generating dataset

```
python borzoi_loader/scripts/generate_dataset_196kb.py \
    chrom_sizes_file[.txt] bigwig_file[.bw] dataset_size[int] out_file[.bed]
```

This script will randomly sample windows by
1. Randomly sampling a chromosome with the probability proportional to the chromosome size
2. Randomly sampling a window within that chromosome, leaving sequence padding space on either end of the window
3. Obtaining the values for this window using `pyBigWig` and summing(default)/averaging these in 32bp bins

The easiest way to limit a dataset to a certain set of chromosomes is to create
a chrom_sizes_file that is limited to those chromosomes.

Alternatively you can specify `--exclude_list_path path_to_file.txt` to exclude certain chromosomes during dataset creation.

A toy output looks something like this
```
chr1	331	371	[20.0, 20.0]
chr1	266	306	[20.0, 20.0]
chr1	646	686	[20.0, 20.0]
chr1	739	779	[20.0, 20.0]
chr1	14	54	[20.0, 20.0]
chr1	596	636	[20.0, 20.0]
...
```

Happy finetuning :-)
