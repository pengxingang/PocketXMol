import numpy as np


def get_points_from_letter(letter, n_points, length=12, height=2):
    """
    Sample point clouds from the shape of a letter.
    """
    
    array = get_2d_points_from_letter(letter, n_points, length)
    # array += np.random.normal(0, 0.1 * length, array.shape)
    
    # Add z coordinate
    z = np.random.uniform(-height / 2, height / 2, n_points)
    points = np.concatenate([array, z[:, None]], axis=-1)
    
    return points
    
def get_2d_points_from_letter(letter, n_points, length):
    
    if letter == 'O':
        radius = length / 2
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        array = np.stack([x, y], axis=-1)
    elif letter == 'M':
        radius = length * 0.75
        left_most = [-radius, 0]
        right_most = [radius, 0]
        left_top = [-radius/2, length]
        right_top = [radius/2, length]
        center = [0, 0.2 * length]
        
        # lines
        n_remain = n_points - 1
        n_per_line = n_remain / 4
        n_per_outer_line = int(np.round(n_per_line * 1.2))
        n_per_inner_line = int(np.round(n_per_line * 0.8))
        array = [
            np.linspace(left_most, left_top, n_per_outer_line, endpoint=False),
            np.linspace(left_top, center, n_per_inner_line, endpoint=False),
            np.linspace(center, right_top, n_per_inner_line, endpoint=False),
            np.linspace(right_top, right_most, n_per_outer_line, endpoint=False),
        ] + [[right_most]]
        array = np.concatenate(array)
        if len(array) < n_points:
            addition = array[np.random.permutation(len(array))[:n_points - len(array)]]
            addition += np.random.randn(*addition.shape) * length * 0.1
            array = np.concatenate([array, addition], axis=0)
        elif len(array) > n_points:
            array = array[np.random.permutation(len(array))[:n_points]]
        array -= np.mean(array, axis=0, keepdims=True) # move to center
    elif letter == 'L':
        center = [0, 0]
        start = [0, length * 1.5]
        end = [length, 0]
        n_left = int(n_points * 0.6)
        n_bottom = n_points - n_left
        array = [
            np.linspace(start, center, n_left, endpoint=False),
            np.linspace(center, end, n_bottom)
        ]
        array = np.concatenate(array, axis=0)
        array -= np.mean(array, axis=0, keepdims=True) # move to center
    elif letter == 'P':
        top = [0, length]
        bottom = [0, -length]
        radius = length / 2 * 1.3
        top_center = [radius*0.2, length - radius]
        # ring
        n_ring = int(n_points * 0.6)
        theta = np.linspace(-np.pi/2*1.1, np.pi/2* 1.1, n_ring)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        array = [
            np.linspace(top, bottom, n_points - n_ring),
            np.stack([x, y], axis=-1) + top_center
        ]
        array = np.concatenate(array, axis=0)
        array -= np.mean(array, axis=0, keepdims=True) # move to center
    elif letter == 'C':
        radius = length / 2
        theta = np.linspace(0.2 * np.pi, (2 - 0.2) * np.pi, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta) * 1.3
        array = np.stack([x, y], axis=-1)
    elif letter == 'K':
        left_top = [0, length]
        left_bottom = [0, -length]
        right_top = [length * 0.8, length*0.7]
        right_bottom = [length * 0.8, -length*0.8]
        center = [0, 0]
        n_quarter = n_points // 4
        n_remain = n_points - n_quarter * 3
        array = [
            np.linspace(left_top, center, n_quarter, endpoint=False),
            np.linspace(right_top, center, n_quarter, endpoint=False),
            np.linspace(right_bottom, center, n_quarter,),
            np.linspace(center, left_bottom, n_remain),
        ]
        array = np.concatenate(array, axis=0)
        array -= np.mean(array, axis=0, keepdims=True) # move to center
    elif letter == 'E':
        radius = length * 0.75
        top_left = [0, radius]
        bottom_left = [0, -radius]
        center = [0, 0]
        top_right = [radius, radius]
        bottom_right = [radius, -radius]
        right_center = [radius*0.8, 0]
        n_radius = int(n_points / 4.8)
        array = [
            np.linspace(top_left, bottom_left, 2 * n_radius),
            np.linspace(top_right, top_left, n_radius, endpoint=False),
            np.linspace(bottom_right, bottom_left, n_radius, endpoint=False),
            np.linspace(right_center, center, n_points - 4 * n_radius, endpoint=False),
        ]
        array = np.concatenate(array, axis=0)
        array -= np.mean(array, axis=0, keepdims=True) # move to center
    elif letter == 'T':
        top_left = [-0.75*length, length]
        top_right = [0.75*length, length]
        top_center = [0, length]
        bottom_center = [0, -length]
        n_top = int(n_points * 0.4)
        n_bottom = n_points - n_top
        array = [
            np.linspace(top_left, top_right, n_top),
            np.linspace(bottom_center, top_center, n_bottom, endpoint=False)
        ]
        array = np.concatenate(array, axis=0)
        array -= np.mean(array, axis=0, keepdims=True) # move to center
    elif letter == 'X':
        radius = length /2 * 1.1
        top_left = [-radius*0.75, radius]
        top_right = [radius*0.75, radius]
        bottom_left = [-radius*0.75, -radius]
        bottom_right = [radius*0.75, -radius]
        n_top = n_points // 2
        n_bottom = n_points - n_top
        array = [
            np.linspace(top_left, bottom_right, n_top),
            np.linspace(top_right, bottom_left, n_bottom)
        ]
        array = np.concatenate(array, axis=0)
        array -= np.mean(array, axis=0, keepdims=True)
    return array
    
    
    
    
if __name__ == '__main__':
    n_points = 40
    # letter = 'O'
    # length = 12
    # height = 5
    # letter = 'M'
    # length = 12
    # height = 5
    letter = 'P'
    length = 12
    height = 5
    points = get_points_from_letter(letter, n_points, length, height)
    
    # make rdmol with points
    from rdkit import Chem
    def points_to_mol(points):
        from rdkit.Chem import AllChem
        from rdkit import Geometry
        # points: (n_atoms, 3)
        n_atoms = len(points)
        rd_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(n_atoms)
        
        # add atoms and coordinates
        for i in range(n_atoms):
            rd_atom = Chem.Atom('C')
            rd_mol.AddAtom(rd_atom)
            rd_coords = Geometry.Point3D(*points[i])
            rd_conf.SetAtomPosition(i, rd_coords)
        rd_mol.AddConformer(rd_conf)
        mol = rd_mol.GetMol()
        return mol
    
    mol = points_to_mol(points)
    Chem.MolToMolFile(mol, f'tmp_{letter}.sdf')
    print('Done')
