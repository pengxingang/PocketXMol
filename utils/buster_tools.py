



# These are from https://github.com/maabuu/posebusters/blob/main/posebusters


"""Module to check intermolecular distances between ligand and protein."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from re import findall

from rdkit.Chem.rdchem import GetPeriodicTable, Mol
from rdkit.Chem.rdchem import Mol
from rdkit.rdBase import LogToPythonLogger
from rdkit import RDLogger
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from rdkit.Chem.Lipinski import HAcceptorSmarts, HDonorSmarts
from rdkit.Chem.rdchem import AtomValenceException, Bond, Conformer, GetPeriodicTable, Mol, RWMol
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
from rdkit.Chem.rdmolfiles import MolFromSmarts
from rdkit.Chem.rdmolops import AddHs, RemoveHs, RemoveStereochemistry, RenumberAtoms, SanitizeMol, AssignStereochemistryFrom3D
from rdkit.Chem.rdMolTransforms import TransformConformer
from rdkit.Chem.inchi import MolFromInchi, MolToInchi
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

_periodic_table = GetPeriodicTable()

"""Protein related functions."""


from typing import Iterable

from rdkit.Chem.rdchem import Atom, Mol

logger = logging.getLogger(__name__)
_inorganic_cofactor_elements = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Cl",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Br",
    "Rb",
    "Mo",
    "Cd",
}
_inorganic_cofactor_ccd_codes = {
    "FES",
    "MOS",
    "PO3",
    "PO4",
    "PPK",
    "SO3",
    "SO4",
    "VO4",
}
_water_ccd_codes = {
    "HOH",
}


def get_atom_type_mask(mol: Mol, ignore_types: Iterable[str]):
    """Get mask for atoms to keep."""
    ignore_types = set(ignore_types)
    unsupported = ignore_types - {"hydrogens", "protein", "organic_cofactors", "inorganic_cofactors", "waters"}
    if unsupported:
        raise ValueError(f"Ignore types {unsupported} not supported.")

    if mol.GetAtomWithIdx(0).GetPDBResidueInfo() is None:
        logger.warning("No PDB information found. Assuming organic molecule.")

    ignore_h = "hydrogens" in ignore_types
    ignore_protein = "protein" in ignore_types
    ignore_org_cof = "organic_cofactors" in ignore_types
    ignore_inorg_cof = "inorganic_cofactors" in ignore_types
    ignore_water = "waters" in ignore_types

    return [
        _keep_atom(a, ignore_h, ignore_protein, ignore_org_cof, ignore_inorg_cof, ignore_water) for a in mol.GetAtoms()
    ]


def _keep_atom(  # noqa: PLR0913, PLR0911
    atom: Atom, ignore_h: bool, ignore_protein: bool, ignore_org_cof: bool, ignore_inorg_cof: bool, ignore_water: bool
):
    """Whether to keep atom for given ignore flags."""
    symbol = atom.GetSymbol()
    if ignore_h and symbol == "H":
        return False

    if ignore_inorg_cof and symbol in _inorganic_cofactor_elements:
        return False

    # if loaded from PDB file, we can use the residue names and the hetero flag
    info = atom.GetPDBResidueInfo()
    if info is None:
        if ignore_org_cof:
            return False
        return True

    is_hetero = info.GetIsHeteroAtom()
    if ignore_protein and not is_hetero:
        return False

    residue_name = info.GetResidueName()
    if ignore_water and residue_name in _water_ccd_codes:
        return False

    if ignore_inorg_cof and residue_name in _inorganic_cofactor_ccd_codes:
        return False

    return True


# inter distance 
def check_intermolecular_distance(  # noqa: PLR0913
    mol_pred: Mol,
    mol_cond: Mol,
    radius_type: str = "vdw",
    radius_scale: float = 1.0,
    clash_cutoff: float = 0.75,
    ignore_types: set[str] = {"hydrogens"},
    max_distance: float = 5.0,
    search_distance: float = 6.0,
):
    """Check that predicted molecule is not too close and not too far away from conditioning molecule.

    Args:
        mol_pred: Predicted molecule (docked ligand) with one conformer.
        mol_cond: Conditioning molecule (protein) with one conformer.
        radius_type: Type of atomic radius to use. Possible values are "vdw" (van der Waals) and "covalent".
            Defaults to "vdw".
        radius_scale: Scaling factor for the atomic radii. Defaults to 0.8.
        clash_cutoff: Threshold for how much the atoms may overlap before a clash is reported. Defaults
            to 0.05.
        ignore_types: Which types of atoms to ignore in mol_cond. Possible values to include are "hydrogens", "protein",
            "organic_cofactors", "inorganic_cofactors", "waters". Defaults to {"hydrogens"}.
        max_distance: Maximum distance (in Angstrom) predicted and conditioning molecule may be apart to be considered
            as valid. Defaults to 5.0.

    Returns:
        PoseBusters results dictionary.
    """
    coords_ligand = mol_pred.GetConformer().GetPositions()
    coords_protein = mol_cond.GetConformer().GetPositions()

    atoms_ligand = np.array([a.GetSymbol() for a in mol_pred.GetAtoms()])
    atoms_protein_all = np.array([a.GetSymbol() for a in mol_cond.GetAtoms()])

    idxs_ligand = np.array([a.GetIdx() for a in mol_pred.GetAtoms()])
    idxs_protein = np.array([a.GetIdx() for a in mol_cond.GetAtoms()])

    mask = [a.GetSymbol() != "H" for a in mol_pred.GetAtoms()]
    coords_ligand = coords_ligand[mask, :]
    atoms_ligand = atoms_ligand[mask]
    mask_ligand_idxs = idxs_ligand[mask]
    if ignore_types:
        mask = get_atom_type_mask(mol_cond, ignore_types)
        coords_protein = coords_protein[mask, :]
        atoms_protein_all = atoms_protein_all[mask]
        mask_protein_idxs = idxs_protein[mask]

    # get radii
    radius_ligand = _get_radii(atoms_ligand, radius_type)
    radius_protein_all = _get_radii(atoms_protein_all, radius_type)

    # select atoms that are close to ligand to check for clash
    distances_all = _pairwise_distance(coords_ligand, coords_protein)
    mask_protein = distances_all.min(axis=0) <= search_distance
    distances = distances_all[:, mask_protein]
    radius_protein = radius_protein_all[mask_protein]
    atoms_protein = atoms_protein_all[mask_protein]
    mask_protein_idxs = mask_protein_idxs[mask_protein]

    radius_sum = radius_ligand[:, None] + radius_protein[None, :]
    relative_distance = distances / radius_sum
    violations = relative_distance < 1 / radius_scale

    if distances.size > 0:
        violations[np.unravel_index(distances.argmin(), distances.shape)] = True  # add smallest distances as info
        violations[np.unravel_index(relative_distance.argmin(), relative_distance.shape)] = True
    violation_ligand, violation_protein = np.where(violations)
    reverse_ligand_idxs = mask_ligand_idxs[violation_ligand]
    reverse_protein_idxs = mask_protein_idxs[violation_protein]

    # collect details around those violations in a dataframe
    details = pd.DataFrame()
    details["ligand_atom_id"] = reverse_ligand_idxs
    details["protein_atom_id"] = reverse_protein_idxs
    details["ligand_element"] = [atoms_ligand[i] for i in violation_ligand]
    details["protein_element"] = [atoms_protein[i] for i in violation_protein]
    details["ligand_vdw"] = [radius_ligand[i] for i in violation_ligand]
    details["protein_vdw"] = [radius_protein[i] for i in violation_protein]
    details["sum_radii"] = details["ligand_vdw"] + details["protein_vdw"]
    details["distance"] = distances[violation_ligand, violation_protein]
    details["sum_radii_scaled"] = details["sum_radii"] * radius_scale
    details["relative_distance"] = details["distance"] / details["sum_radii_scaled"]
    details["clash"] = details["relative_distance"] < clash_cutoff

    results = {
        "smallest_distance": details["distance"].min(),
        "not_too_far_away": details["distance"].min() <= max_distance,
        "num_pairwise_clashes": details["clash"].sum(),
        "no_clashes": not details["clash"].any(),
    }

    # add most extreme values to results table
    i = np.argmin(details["relative_distance"]) if len(details) > 0 else None
    most_extreme = {"most_extreme_" + c: details.loc[i][str(c)] if i is not None else pd.NA for c in details.columns}
    results = {**results, **most_extreme}

    return {"results": results, "details": details}



def _pairwise_distance(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def _get_radii(atoms: np.ndarray, radius_type: str):
    if radius_type == "vdw":
        return np.array([_periodic_table.GetRvdw(a) for a in atoms])
    elif radius_type == "covalent":
        return np.array([_periodic_table.GetRcovalent(a) for a in atoms])
    else:
        raise ValueError(f"Unknown radius type {radius_type}. Valid values are 'vdw' and 'covalent'.")


"""Module to check identity of docked and crystal ligand."""




LogToPythonLogger()
logger = logging.getLogger(__name__)


def check_identity(mol_pred: Mol, mol_true: Mol, inchi_options: str = ""):
    """Check two molecules are identical (docking relevant identity).

    Args:
        mol_pred: Predicted molecule (docked ligand).
        mol_true:Ground truth molecule (crystal ligand) with a conformer.
        inchi_options: String of options to pass to the InChI module. Defaults to "".

    Returns:
        PoseBusters results dictionary.
    """
    # generate inchis
    inchi_crystal = standardize_and_get_inchi(mol_true, options=inchi_options)
    inchi_docked = standardize_and_get_inchi(mol_pred, options=inchi_options)

    # check inchis are valid
    inchi_crystal_valid = is_valid_inchi(inchi_crystal)
    inchi_docked_valid = is_valid_inchi(inchi_docked)

    # compare inchis
    if inchi_crystal_valid and inchi_docked_valid:
        inchi_comparison = _compare_inchis(inchi_docked, inchi_crystal)
    else:
        inchi_comparison = {}

    results = {
        "inchi_crystal_valid": inchi_crystal_valid,
        "inchi_docked_valid": inchi_docked_valid,
        "inchi_crystal": inchi_crystal,
        "inchi_docked": inchi_docked,
        **inchi_comparison,
    }

    return {"results": results}


# isotopic layer /i (for isotopes) not present if no isotopic information in molecule
# fixed hydrogen layer /f (for tautomers) not part of standard inchi string
# reconnected layer /r (for metal ions) not part of standard inchi string
standard_layers = ["=", "/", "/c", "/h", "/q", "/p", "/t", "/b", "/m", "/s"]
layer_names = {
    "=": "inchi_version",
    "/": "formula",
    "/c": "connections",
    "/h": "hydrogens",
    "/q": "net_charge",
    "/p": "protons",
    "/b": "stereo_dbond",  # double bond (Z/E) stereochemistry
    "/t": "stereo_sp3",  # tetrahderal stereochemistry
    "/m": "stereo_sp3_inverted",
    "/s": "stereo_type",
    "/i": "isotopic",
}
stereo_all_layers = ["stereo_dbond", "stereo_sp3", "stereo_sp3_inverted", "stereo_type"]
stereo_tetrahedral_layers = ["stereo_sp3", "stereo_sp3_inverted", "stereo_type"]


def _compare_inchis(inchi_true: str, inchi_pred: str, layers: list[str] = standard_layers):
    results = {}

    # fast return when overall InChI is the same
    if inchi_true == inchi_pred:
        results["inchi_overall"] = True
        for layer in layers:
            results[layer_names[layer]] = True
            results["stereo_tetrahedral"] = True
            results["stereo"] = True
        return results

    # otherwise comparison by layer
    results["inchi_overall"] = False
    layers_true = split_inchi(inchi_true)
    layers_pred = split_inchi(inchi_pred)
    assert "/" in layers_true, "Molecular formula layer missing from InChI string"
    for layer in layers:
        name = layer_names[layer]
        if (layer not in layers_true) or (layer not in layers_pred):
            results[name] = (layer not in layers_true) and (layer not in layers_pred)
        else:
            results[name] = layers_true[layer] == layers_pred[layer]

    # combine stereo layers (pass if not present)
    results["stereo_tetrahedral"] = all(results.get(name, True) for name in stereo_tetrahedral_layers)
    results["stereo"] = all(results.get(name, True) for name in stereo_all_layers)

    return results


def standardize_and_get_inchi(mol: Mol, options: str = "", log_level=None, warnings_as_errors=False):
    """Return InChI after standardising molecule and inferring stereo from coordinates."""
    mol = deepcopy(mol)
    mol = assert_sanity(mol)

    # standardise molecule
    mol = remove_isotopic_info(mol)

    # assign stereo from 3D coordinates only if 3D coordinates are present
    has_pose = mol.GetNumConformers() > 0
    if has_pose:
        RemoveStereochemistry(mol)

    mol = RemoveHs(mol)
    try:
        mol = neutralize_atoms(mol)
    except AtomValenceException:
        logger.warning("Failed to neutralize molecule. Using uncharger. InChI check might fail.")
        mol = Uncharger().uncharge(mol) 
    mol = add_stereo_hydrogens(mol)

    if has_pose:
        AssignStereochemistryFrom3D(mol, replaceExistingTags=True)

    with CaptureLogger():
        inchi = MolToInchi(mol, options=options, logLevel=log_level, treatWarningAsError=warnings_as_errors)

    return inchi


def is_valid_inchi(inchi: str) -> bool:
    """Check that InChI can be parsed and sanitization does not fail."""
    try:
        mol = MolFromInchi(inchi)
        assert_sanity(mol)
        assert mol is not None
        return True
    except Exception:
        return False


def split_inchi(inchi: str):
    """Split the standard InChI string without isotopic info into its layers."""
    if not inchi.startswith("InChI="):
        raise ValueError("InChI string must start with 'InChI='")

    # inchi always InChi=1S/...formula.../...layer.../...layer.../...layer...
    version = inchi[6:].split(r"/", 1)[0]
    layers = findall(r"(?=.*)(\/[a-z]{0,1})(.*?)(?=\/.*|$)", inchi[6:])

    inchi_parts = {"=": version}
    for prefix, layer in layers:
        # standard inchi strings without isotopic info have each layer no more than once
        assert prefix not in inchi_parts, f"Layer {prefix} more than once!"
        inchi_parts[prefix] = layer

    return inchi_parts


###### mol tools

def assert_sanity(mol: Mol):
    """Check that RDKit sanitization does not fail."""
    flags = SanitizeMol(mol)
    assert flags == 0, f"Sanitization failed with flags {flags}"
    return mol


def remove_isotopic_info(mol: Mol):
    """Remove isotopic information from molecule."""
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


def neutralize_atoms(mol: Mol):
    """Add and remove hydrogens to neutralize charges ignoring overall charge."""
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules
    # stronger than rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger
    pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def add_stereo_hydrogens(mol: Mol):
    """Add all hydrogens but those on primary ketimine."""
    exclude = {match[1] for match in mol.GetSubstructMatches(MolFromSmarts("[CX3]=[NH1]"))}
    atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1 if a.GetIdx() not in exclude]
    mol = AddHs(mol, onlyOnAtoms=atoms, addCoords=True)
    return mol




"""Logging utilities."""

import os
import sys

import rdkit
from rdkit import rdBase

# redirect logs to Python logger
rdBase.LogToPythonLogger()


# https://github.com/rdkit/rdkit/discussions/5435
class CaptureLogger(logging.Handler):
    """Helper class that captures Python logger output."""

    def __init__(self, module=None):
        """Initialize logger."""
        super().__init__(level=logging.NOTSET)
        self.logs = {}
        self.devnull = open(os.devnull, "w")
        rdkit.log_handler.setStream(self.devnull)
        rdkit.logger.addHandler(self)

    def __enter__(self):
        """Enter context manager."""
        return self.logs

    def __exit__(self, *args):
        """Exit context manager."""
        self.release()

    def handle(self, record):
        """Handle log record."""
        key = record.levelname
        val = self.format(record)
        self.logs[key] = self.logs.get(key, "") + val
        return False

    def release(self):
        """Release logger."""
        rdkit.log_handler.setStream(sys.stderr)
        rdkit.logger.removeHandler(self)
        self.devnull.close()
        return self.logs