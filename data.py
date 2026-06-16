from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SEQ_LENGTH = 12288

#BASE_DATA_PATH = "/home/valentin/master/ml4rg/clusterhome/project03/toga_13_species_updated/"
BASE_DATA_PATH = "/data/ceph/hdd/project/node_07/ml4rg_students/2026/project03/toga_13_species_updated/"
SPECIES_TSVS: Mapping[str, str] = {
    "bat": BASE_DATA_PATH + "saluki_bat.hg38.HLrouAeg4.tsv",
    "cattle": BASE_DATA_PATH + "saluki_cattle.hg38.bosTau9.tsv",
    "dog": BASE_DATA_PATH + "saluki_dog.hg38.canFam6.tsv",
    "hare": BASE_DATA_PATH + "saluki_hare.hg38.HLlepYar1.tsv",
    "horse": BASE_DATA_PATH + "saluki_horse.hg38.HLequCaba4.tsv",
    "human": BASE_DATA_PATH + "saluki_human.hg38.hg38.tsv",
    "marmoset": BASE_DATA_PATH + "saluki_marmoset.hg38.HLcalJacc6.tsv",
    "mouse": BASE_DATA_PATH + "saluki_mouse.mm10.mm10.tsv",
    "pig": BASE_DATA_PATH + "saluki_pig.hg38.HLsusScro12.tsv",
    "rabbit": BASE_DATA_PATH + "saluki_rabbit.hg38.HLoryCuni5.tsv",
    "rat": BASE_DATA_PATH + "saluki_rat.hg38.HLratNor8.tsv",
    "rhesus": BASE_DATA_PATH + "saluki_rhesus.hg38.HLmacMula11.tsv",
    "whale": BASE_DATA_PATH + "saluki_whale.hg38.HLmegNova3.tsv",
}
# todo: find better species order for project extension
SPECIES_ORDER: Tuple[str, str] = (
    "human", "mouse"
    #,"bat", "cattle", "dog", "hare", "horse", "marmoset",  "pig", "rabbit", "rat", "rhesus", "whale"
)

# todo: read this in as  an argument from startup shall script / sbatch
SPLIT_TO_FOLDS: Mapping[str, Tuple[int, ...]] = {
    "test": (0,),
    "valid": (1,),
    "train": tuple(range(2, 10)),
}

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "gene",
    "fold",
    "half_life",
    "seq",
    "frame",
    "splice",
)


def _read_saluki_tsv(tsv_path: str | Path) -> pd.DataFrame:
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Saluki TSV not found: {path}")

    df = pd.read_csv(path, sep="\t", usecols=list(REQUIRED_COLUMNS))
    return df


def _split_dataframe(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLIT_TO_FOLDS:
        valid = ", ".join(sorted(SPLIT_TO_FOLDS))
        raise ValueError(f"Unknown split {split!r}; expected one of: {valid}")

    split_df = df[df["fold"].isin(SPLIT_TO_FOLDS[split])].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split {split!r}")

    return split_df.reset_index(drop=True)


def _encode_sequence(seq: str, frame: str, splice: str, seq_length: int) -> torch.Tensor:
    length = len(seq)
    if len(frame) != length or len(splice) != length:
        raise ValueError(
            "Sequence, frame, and splice tracks must have the same length "
            f"(got {length}, {len(frame)}, {len(splice)})"
        )
    if length > seq_length:
        raise ValueError(f"Sequence length {length} exceeds configured length {seq_length}")

    encoded = np.zeros((seq_length, 6), dtype=np.float32)

    # Channels 0-3 are A/C/G/T. Unknown symbols remain all-zero.
    for channel, base in enumerate("ACGT"):
        encoded[:length, channel] = np.fromiter(
            (nt == base for nt in seq),
            dtype=np.float32,
            count=length,
        )

    encoded[:length, 4] = np.fromiter(
        (value == "X" for value in frame),
        dtype=np.float32,
        count=length,
    )
    encoded[:length, 5] = np.fromiter(
        (value == "X" for value in splice),
        dtype=np.float32,
        count=length,
    )

    return torch.from_numpy(encoded)


class SalukiTsvDataset(Dataset):
    """PyTorch dataset for Saluki human/mouse TSV measurement files."""

    def __init__(
        self,
        tsv_path: str | Path,
        split: str,
        seq_length: int = SEQ_LENGTH,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        self.tsv_path = str(tsv_path)
        self.split = split
        self.seq_length = seq_length

        full_df = _read_saluki_tsv(tsv_path) if dataframe is None else dataframe
        self.dataframe = _split_dataframe(full_df, split)


    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        x = _encode_sequence(
            str(row["seq"]),
            str(row["frame"]),
            str(row["splice"]),
            self.seq_length,
        )

        target = float(row["half_life"])
        y = torch.tensor([target], dtype=torch.float32)
        return x, y


def _make_dataloader(
    dataset: Dataset,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if split == "train" else False,
    )


def create_saluki_dataloaders(
    species_tsvs: Optional[Mapping[str, str | Path]] = None,
    species_order: Sequence[str] = SPECIES_ORDER,
    batch_size: int = 64,
    seq_length: int = SEQ_LENGTH,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    include_test: bool = True,
) -> Tuple[List[DataLoader], List[DataLoader], Optional[List[DataLoader]]]:
    """Create train/valid/test dataloaders ordered by species.

    The returned train and validation loader lists can be passed directly to
    ``SalukiTrainer``. By default, species index 0 is human and index 1 is mouse.
    """

    paths: Dict[str, str | Path] = dict(SPECIES_TSVS if species_tsvs is None else species_tsvs)
    train_loaders: List[DataLoader] = []
    valid_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []
    # print(f'DEBUG {paths}')
    for species in species_order:
        if species not in paths:
            raise ValueError(f"No TSV path configured for species {species!r}")

        full_df = _read_saluki_tsv(paths[species])

        for split, target_list in (
            ("train", train_loaders),
            ("valid", valid_loaders),
            ("test", test_loaders),
        ):
            if split == "test" and not include_test:
                continue
            dataset = SalukiTsvDataset(
                paths[species],
                split=split,
                seq_length=seq_length,
                dataframe=full_df,
            )
            target_list.append(
                _make_dataloader(
                    dataset,
                    split=split,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                )
            )
    if include_test:
        return train_loaders, valid_loaders, test_loaders
    else:
        return train_loaders, valid_loaders
