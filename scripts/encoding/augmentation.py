# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_data_augmentation.ipynb (unless otherwise specified).

__all__ = ['randomize_smiles', 'precursor_permutation_given_index', 'randomize_rxn', 'do_randomizations_on_df',
           'do_random_permutations_on_df']

# Cell
import random
import math
import numpy as np
import pandas as pd
from rdkit import Chem

# Cell
def randomize_smiles(smiles, random_type="rotated", isomericSmiles=True):
    """
    From: https://github.com/undeadpixel/reinvent-randomized and https://github.com/GLambard/SMILES-X
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted, rotated) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=isomericSmiles)
    elif random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=isomericSmiles)
    elif random_type == 'rotated':
        n_atoms = mol.GetNumAtoms()
        rotation_index = random.randint(0, n_atoms-1)
        atoms = list(range(n_atoms))
        new_atoms_order = (atoms[rotation_index%len(atoms):]+atoms[:rotation_index%len(atoms)])
        rotated_mol = Chem.RenumberAtoms(mol,new_atoms_order)
        return Chem.MolToSmiles(rotated_mol, canonical=False, isomericSmiles=isomericSmiles)
    raise ValueError("Type '{}' is not valid".format(random_type))

# Cell
from itertools import permutations

def precursor_permutation_given_index(precursor_list, permutation_index):
    """
    Return permutation of list given index. Inspired by
    https://stackoverflow.com/questions/5602488/random-picks-from-permutation-generator.
    :param precursor_list: List of molecules
    :param permutation_index: Permutation index
    :return : Permuted list
    """

    precursor_list = precursor_list[:]
    for i in range(len(precursor_list)-1):
        permutation_index, j = divmod(permutation_index, len(precursor_list)-i)
        precursor_list[i], precursor_list[i+j] = precursor_list[i+j], precursor_list[i]
    return precursor_list

def randomize_rxn(rxn, random_type):
    """
    Split reaction into precursors and products, then randomize all molecules.
    """
    precursors, product = rxn.split('>>')
    precursors_list = precursors.split('.')

    randomized_precursors = [randomize_smiles(precursor, random_type) for precursor in precursors_list]
    randomized_product = randomize_smiles(product, random_type)
    return f"{'.'.join(randomized_precursors)}>>{randomized_product}"


def do_randomizations_on_df(df, n_randomizations=1, random_type='rotated', seed=42):
    """
    Randomize all molecule SMILES of the reactions in a dataframe.
    Expected to have column 'text' with the reactions and 'label' with the property to predict.
    """
    new_texts = []
    new_labels = []
    random.seed(seed)

    for i, row in df.iterrows():
        if random_type != '':
            randomized_rxns = [randomize_rxn(row['text'], random_type=random_type) for i in range(n_randomizations)]
        new_texts.extend(randomized_rxns)
        new_labels.extend([row['labels']]*len(randomized_rxns))
    return pd.DataFrame({'text': new_texts, 'labels': new_labels})


def do_random_permutations_on_df(df, n_permutations=1, fixed=False, random_type='', seed=42):
    """
    Generate `n_permutations` permutations of the precursors per reaction in a dataframe.
    Expected to have column 'text' with the reactions and 'label' with the property to predict.

    """

    new_texts = []
    new_labels = []

    for i, row in df.iterrows():

        precursors, product = row['text'].split('>>')
        precursors_list = precursors.split('.')
        if fixed:
            random.seed(seed)
        else:
            random.seed(i+seed)
        total_permutations = range(math.factorial(len(precursors_list)))

        permutation_indices = random.sample(total_permutations, n_permutations)
        permuted_rxns = []

        for idx in permutation_indices:
            permuted_precursors = precursor_permutation_given_index(precursors_list, idx)
            permuted_rxns.append(f"{'.'.join(permuted_precursors)}>>{product}")
        if random_type != '':
            permuted_rxns = [randomize_rxn(rxn, random_type=random_type) for rxn in permuted_rxns]
        new_texts.extend(permuted_rxns)
        new_labels.extend([row['labels']]*len(permuted_rxns))
    return pd.DataFrame({'text': new_texts, 'labels': new_labels})
