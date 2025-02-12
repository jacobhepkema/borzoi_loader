import ast
import torch
from enformer_pytorch import GenomeIntervalDataset


class GenomeDataIntervalDataset(GenomeIntervalDataset):
    def __init__(self, all_chroms, extend_seq=16 * 32, *args, **kwargs):
        """
        Argument all_chroms is a list of chromosomes in the dataset.
        E.g. ['chr1', 'chr2', ...].

        Argument extend_seq specifies how many basepairs to extend the sequence
        on either side by. Default = 16 * 32, which is compatible with Borzoi.
        Note that you might need to specify the data generation process with
        generate_dataset.py as well so that the sampled regions always have
        enough padding along the sides of the region to allow for the extend_seq
        amount. Use argument --padding in those scripts for this - the default
        compatible value is 16 * 32 for that one too.
        """
        super(GenomeDataIntervalDataset, self).__init__(*args, **kwargs)
        self.chr_to_int = dict(zip(all_chroms, range(len(all_chroms))))
        self.int_to_chr = {v: k for k, v in self.chr_to_int.items()}
        self.extend_seq = extend_seq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, target = (
            interval[0], interval[1], interval[2], interval[3])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        target = ast.literal_eval(target)
        target = torch.tensor(target).unsqueeze(-1)

        # This will break if you do not leave enough space for the padding:
        ext_start = start - self.extend_seq
        ext_end = end + self.extend_seq

        sequence = self.fasta(chr_name, ext_start, ext_end,
                              return_augs=self.return_augs)
        loc = torch.tensor([self.chr_to_int[chr_name], start, end])
        return sequence, target, loc

    @staticmethod
    def check_tensor_dtype(tensor):
        if tensor.dtype == torch.float16:
            return "fp16"
        elif tensor.dtype == torch.bfloat16:
            return "bf16"
        else:
            return str(tensor.dtype)
