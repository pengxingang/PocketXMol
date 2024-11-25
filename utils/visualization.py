

from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole



def add_index(mol):
    mol = deepcopy(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def compute_2dcoords(mol):
    mol = deepcopy(mol)
    AllChem.Compute2DCoords(mol)
    return mol

def show(mol, size=(400, 300), **kwargs):
    return Draw.MolToImage(mol, size=size, **kwargs)

def show2D(mol, size=(400, 300), with_index=True, **kwargs):
    if not with_index:
        return show(compute_2dcoords(mol), size=size, **kwargs)
    return show(compute_2dcoords(add_index(mol)), size=size, **kwargs)

def show3D(x):
    print(Chem.MolToSmiles(x))
    IPythonConsole.drawMol3D(x)
    return x

def show_mols(mols, n=8, subImgSize=(250,200), **kwargs):
    mols2d = [Chem.MolFromSmiles(Chem.MolToSmiles(x)) for x in mols]
    return Draw.MolsToGridImage(mols2d, molsPerRow=n, subImgSize=subImgSize, **kwargs)