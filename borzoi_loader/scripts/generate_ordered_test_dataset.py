import argparse
import pyBigWig
import pandas as pd
import numpy as np

from borzoi_loader.utils import get_chrom_sizes, sum_bin, avg_bin, get_bw_signal


def main(chrom_sizes_file, bw_path, output_file, chrom,
         n_bins=6144, bin_size=32, padding=163840, use_sum=True):

    SEQ_LEN = n_bins * bin_size

    bw_file = pyBigWig.open(bw_path)
    chrom_sizes = get_chrom_sizes(chrom_sizes_file)
    assert chrom in chrom_sizes.keys()
    chrom_size = chrom_sizes[chrom]
    int_start = padding
    intervals = [(chrom, s, s + SEQ_LEN) for s in range(int_start, chrom_size - padding - SEQ_LEN, SEQ_LEN)]

    with open(output_file, 'w') as f:
        for chrom, start, end in intervals:
            # Get the signal
            values = get_bw_signal(bw_file, chrom, start, end, SEQ_LEN)
            if use_sum:
                binned_values = sum_bin(values, n_bins)
            else:
                binned_values = avg_bin(values, n_bins)
            # Do not include regions where all values are 0
            row_data = f"{chrom}\t{start}\t{end}\t[{','.join(map(str, binned_values))}]\n"
            f.write(row_data)
    print(f"Dataset saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate an ordered dataset for test prediction windows for a given chromosome.")
    parser.add_argument('chrom_sizes_file', type=str,
                        help='Path to the chromosome sizes file')
    parser.add_argument('bw_path', type=str,
                        help='Path to the .bigWig / .bw file')
    parser.add_argument('output_file', type=str,
                        help='Path to the output file')
    parser.add_argument('chrom', type=str,
                        help='Chromosome to create the test dataset for')
    parser.add_argument('--n_bins', type=int, default=6144,
                        help='Number of bins in prediction window (default: 6144)')
    parser.add_argument('--bin_size', type=int, default=32,
                        help='Size of each bin (default: 32)')
    parser.add_argument('--padding', type=int, default=163840,
                        help='Amount of padding to add to each side of the region later, in basepairs (default: 163840)')
    parser.add_argument('--use_sum', action='store_true',
                        help='Sum within each bin instead of average')

    # TODO add option to ignore blacklisted or masked regions
    args = parser.parse_args()
    main(*vars(args).values())
