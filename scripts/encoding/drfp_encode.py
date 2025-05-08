import os
import pickle
import click
import logging
import multiprocessing
from functools import partial
from typing import Iterable

from drfp import DrfpEncoder

import numpy as np


def encode(smiles: Iterable, length: int = 2048, radius: int = 3) -> np.ndarray:
    return DrfpEncoder.encode(
        smiles,
        n_folded_length=length,
        radius=radius,
        rings=True,
    )


def encode_dataset(smiles: Iterable, length: int, radius: int) -> np.ndarray:
    """Encode the reaction SMILES to drfp"""
    cpu_count = multiprocessing.cpu_count()
    k, m = divmod(len(smiles), cpu_count)
    smiles_chunks = (smiles[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(cpu_count))
    results = []
    with multiprocessing.Pool(cpu_count) as p:
        results = p.map(partial(encode, length=length, radius=radius), smiles_chunks)
    return np.array([item for s in results for item in s])


def add_split_to_filepath(filepath: str) -> str:
    name, ext = os.path.splitext(filepath)
    return f"{name}{ext}"


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--length", type=int, default=2048)
@click.option("--radius", type=int, default=3)
def main(input_filepath, output_filepath, length, radius):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    split_filepath = add_split_to_filepath(output_filepath)


    try:
        with open(input_filepath, 'rb') as f:
            data = pickle.load(f)

        all_smiles = []
        all_labels = []
        for key, value in data.items():
            smiles = value['smiles']
            label = value['label'][0]
            all_smiles.append(smiles)
            all_labels.append(label)

        logger.info("generating drfp fingerprints")
        X = encode_dataset(all_smiles, length, radius)
        y = np.array(all_labels)

        logger.info(f"pickling data set to {split_filepath}")
        with open(split_filepath, "wb+") as f:
            pickle.dump((X, y, np.array(all_smiles)), f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    

'''
root@gpu7-wx ~/wx_projects/SF_yield main* 3m 37s
drfp-environment ❯ python scripts/encoding/drfp_encode.py data/SF640_A_B_C_lbl.pkl data/SF640_A_B_C_lbl-2048-3-true.pkl

'''