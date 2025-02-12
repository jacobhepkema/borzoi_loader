def write_bedgraph(outfile, values, start, chrom, bin_size=128):
    """
    Writes a bedGraph file to outfile. Assumes all values
    border one another, with bin_size distance, and that
    all values are on one chromosome.
    """
    with open(outfile, 'w') as f:
        for i, value in enumerate(list(values)):
            f.write(f'{chrom}\t{start + i * 128}\t{start + (i + 1) * 128}\t{value}\n')
