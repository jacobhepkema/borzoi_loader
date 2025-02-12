#!/usr/bin/env python3

# Many functions adapted from Adapted from
# https://github.com/wconnell/enformer-finetune/blob/c2145a628efcb91b932cc063a658e4a994bc4baa/eft/preprocess.py

import numpy as np


def get_chrom_sizes(chrom_sizes_file) -> dict:
    """
    Get chromosome sizes from a file.
    """
    chrom_sizes = {}
    with open(chrom_sizes_file, "r") as f:
        for line in f:
            fields = line.split()
            chrom, size = fields[0], int(fields[1])
            if 'MT' not in chrom and 'KI' not in chrom and 'GL' not in chrom:
                chrom_sizes[chrom] = size
    return chrom_sizes


def avg_bin(array, n_bins):
    """
    Averages array values in n_bins.
    """
    splitted = np.array_split(array, n_bins)
    return [np.mean(a) for a in splitted]


def sum_bin(array, n_bins):
    """
    Sums array values in n_bins.
    """
    splitted = np.array_split(array, n_bins)
    return [np.sum(a) for a in splitted]


def get_bw_signal(bw_file, chrom, start, end, SEQ_LEN=524288 - (16 * 32 * 2)):
    """
    Get signal from a bigwig file.
    If the chromosome is not found, return a list of np.nan.

    Arguments:
    - bw_file: pyBigWig file
    - chrom: chromosome
    - start: start position
    - end: end position
    - SEQ_LEN: length of the sequence (default: 114688)
    """
    center = (start + end) // 2
    start = center - (SEQ_LEN // 2)
    end = center + (SEQ_LEN // 2)
    try:
        values = bw_file.values(chrom, start, end)
        values = np.nan_to_num(values).tolist()
    except Exception:
        values = [np.nan] * SEQ_LEN
    return values


def random_region(chrom_sizes,
                  bw_file, p=None,
                  SEQ_LEN=524288 - (16 * 32 * 2),
                  padding=512):
    """
    Get a random region from the genome. Returns the chrom, start, end, and bw
    values corresponding to the region.

    Note: SEQ_LEN is the length of the sequence where the padding has already
    been subtracted.

    Defaults are compatible with Borzoi, where each prediction window is trimmed
    with 32 * 128 bp on either side of the prediction window.
    """
    chrom = np.random.choice(list(chrom_sizes.keys()), p=p)
    # Make sure that the start position is at least padding away from the
    # beginning / end of the chromosome
    start = np.random.randint(0 + padding,
                              chrom_sizes[chrom] - SEQ_LEN - padding)
    end = start + SEQ_LEN
    values = get_bw_signal(bw_file, chrom, start, end, SEQ_LEN)
    return chrom, start, end, values
