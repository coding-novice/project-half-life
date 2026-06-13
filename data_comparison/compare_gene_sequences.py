import pandas as pd


our_data = "saluki_human.hg38_only_seq.tsv"

utr5_file = "Homo_sapiens.GRCh38.83.chosenTranscript.5pUTRs_formatted.tsv"
orf_file = "Homo_sapiens.GRCh38.83.chosenTranscript.ORFs_formatted.tsv"
utr3_file = "Homo_sapiens.GRCh38.83.chosenTranscript.3pUTRs_formatted.tsv"

gene_col = "gene"
seq_col = "seq"

output = "first_difference_orf_boundary_check.tsv"

stop_codons = {"TAA", "TAG", "TGA"}


def read_gene_seq_file(path):
    df = pd.read_csv(path, sep="\t", dtype=str)

    df[gene_col] = df[gene_col].str.strip()
    df[seq_col] = (
        df[seq_col]
        .str.strip()
        .str.upper()
        .str.replace("U", "T", regex=False)
        .str.replace(" ", "", regex=False)
    )

    return dict(zip(df[gene_col], df[seq_col]))


def first_difference(seq1, seq2):
    min_len = min(len(seq1), len(seq2))

    for i in range(min_len):
        if seq1[i] != seq2[i]:
            return i

    if len(seq1) != len(seq2):
        return min_len

    return -1


def main():
    ours = read_gene_seq_file(our_data)
    utr5 = read_gene_seq_file(utr5_file)
    orf = read_gene_seq_file(orf_file)
    utr3 = read_gene_seq_file(utr3_file)

    rows = []
    total = len(ours)

    for n, gene in enumerate(ours.keys(), start=1):
        if n % 500 == 0 or n == total:
            print(f"Genes {n}/{total} finished")

        if gene not in utr5 or gene not in orf or gene not in utr3:
            rows.append({
                "gene": gene,
                "status": "missing_component",
                "length_ours": len(ours[gene]),
                "length_reference": "",
                "length_difference": "",
                "first_difference_1_based": "",
                "orf_end_1_based": "",
                "first_difference_is_right_after_orf": "",
                "triplet_at_difference_in_ours": "",
                "triplet_is_stop_codon": "",
            })
            continue

        seq_ours = ours[gene]
        seq_reference = utr5[gene] + orf[gene] + utr3[gene]

        # 0-based index of the first nucleotide after ORF in the reference
        orf_end_0_based = len(utr5[gene]) + len(orf[gene])

        first_diff_0_based = first_difference(seq_ours, seq_reference)

        if first_diff_0_based == -1:
            status = "identical"
            first_diff_1_based = ""
            is_right_after_orf = ""
            triplet = ""
            triplet_is_stop = ""
        else:
            first_diff_1_based = first_diff_0_based + 1
            is_right_after_orf = first_diff_0_based == orf_end_0_based
            triplet = seq_ours[first_diff_0_based:first_diff_0_based + 3]
            triplet_is_stop = triplet in stop_codons

            if is_right_after_orf:
                status = "first_difference_right_after_orf"
            else:
                status = "first_difference_elsewhere"

        rows.append({
            "gene": gene,
            "status": status,
            "length_ours": len(seq_ours),
            "length_reference": len(seq_reference),
            "length_difference": len(seq_ours) - len(seq_reference),
            "first_difference_1_based": first_diff_1_based,
            "orf_end_1_based": orf_end_0_based + 1,
            "first_difference_is_right_after_orf": is_right_after_orf,
            "triplet_at_difference_in_ours": triplet,
            "triplet_is_stop_codon": triplet_is_stop,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output, sep="\t", index=False)

    print("\nDone.")
    print(out["status"].value_counts())
    print("\nSaved:", output)


if __name__ == "__main__":
    main()