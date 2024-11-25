# from cmath import e
from copy import deepcopy
from typing import Optional, Union
import os
import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
# from torch_geometric.data import Batch
# from tqdm.auto import tqdm
# from sklearn.cluster import DBSCAN, KMeans, OPTICS

# from .common import split_tensor_by_batch, concat_tensors_to_batch
# from utils.noise import *

ALT_KEYS = {'node':'node_type', 'pos':'node_pos', 'halfedge':'halfedge_type'}


# def overwrite_pos_batch(over_cfg, batch, i_repeat, i_batch, batch_size):
#     """
#     Overwrite the pos of the batch with the given configuration, not for individual data
#     """
#     if over_cfg['strategy'] == 'linking_unfixed':
#         pos_root = over_cfg['pos_root']
#         lv = over_cfg['lv']
#         # determine the sdf dir names
#         i_start = i_batch * batch_size
#         i_end = i_start + len(batch)
#         key_batch = batch['key']
#         sep_names = [key.split(';')[-1].replace('/', '_sep_') for key in key_batch]
#         dir_names = [f'{index}-{sep_name}' for index, sep_name in zip(range(i_start, i_end), sep_names)]
#         sdf_paths = [os.path.join(pos_root, f'lv{lv}', dir_name, f'repeat_{i_repeat}.sdf') for dir_name in dir_names]
#         # read the sdf
#         mol_list = [Chem.MolFromMolFile(sdf_path, sanitize=False) for sdf_path in sdf_paths]
#         conf_list = [mol.GetConformer(0).GetPositions() for mol in mol_list]
#         new_pos = np.concatenate(conf_list, axis=0)
        
#         # overwrite the pos
#         node_pos = batch['node_pos']
#         assert node_pos.shape == new_pos.shape, f'node_pos.shape: {node_pos.shape}, new_pos.shape: {new_pos.shape}'
#         batch['node_pos'] = torch.tensor(new_pos, dtype=node_pos.dtype, device=node_pos.device)
#         if batch['pocket_center'].shape[0] > 0:
#             pocket_center = batch['pocket_center']
#             batch['node_pos'] = batch['node_pos'] - pocket_center[batch['node_type_batch']]
        
        


def add_info_to_dict(data_dict, data, mode):
    assert mode in ['in', 'out', 'gt', 'raw_out'], f'Unknown mode: {mode}'
    for key in ['node', 'pos', 'halfedge']:
        if mode == 'in':
            this = data[key+'_in']
        elif mode == 'out':
            this = data[ALT_KEYS[key]]
        elif mode == 'gt':
            if 'gt_' + ALT_KEYS[key] in data:
                this = data['gt_' + ALT_KEYS[key]]
            else:
                continue
        else: # raw_out
            if key != 'pos':
                this = data['pred_' + key].argmax(dim=-1)
            else:
                this = data['pred_' + key]
        data_dict[key].append(this.detach().cpu())

def sample_loop3(batch, model, noiser, device=None, is_ar='', off_tqdm=False):
    # # save initial batch and gt mol seperations
    traj_dict = {'node': [], 'pos': [], 'halfedge': []}
    cfd_traj = []
    mol_parts = get_mol_parts_linking(batch, device=device)
    for data_mol in [batch] + mol_parts:
        add_info_to_dict(traj_dict, data_mol, 'gt')

    # # sample loop
    in_dict = {'node': [], 'pos': [], 'halfedge': []}
    out_dict = {'node': [], 'pos': [], 'halfedge': []}
    raw_dict = {'node': [], 'pos': [], 'halfedge': []}
    
    step_ar = 0
    while True:
        for step in tqdm(noiser.steps_loop(add_last=False), desc='Sampling', total=noiser.num_steps, disable=off_tqdm):
            with torch.no_grad():
                # # add noise as the input of the step
                batch = noiser(batch, step)
                add_info_to_dict(in_dict, batch, 'in')
                add_info_to_dict(traj_dict, batch, 'in')
                
                # # denoise and correct output in output space
                outputs = model(batch) 
                batch.update({'step': step})
                batch = noiser.outputs2batch(batch, outputs)  # correct the denoised, as the next step start mol
                add_info_to_dict(out_dict, batch, 'out')
                add_info_to_dict(traj_dict, batch, 'out')
                add_info_to_dict(raw_dict, outputs, 'raw_out')
                cfd_traj.append(outputs['confidence_pos'].detach().flatten())


        if is_ar.startswith('ar'):
            if is_ar == 'ar':
                batch = noiser.outputs2batch_ar(batch, outputs, step_ar, cfd_traj)
            elif is_ar == 'ar2':
                batch = noiser.outputs2batch_ar2(batch, outputs)
            add_info_to_dict(out_dict, batch, 'out')
            add_info_to_dict(traj_dict, batch, 'out')
            add_info_to_dict(raw_dict, outputs, 'raw_out')
            if batch['node_p2'].shape[0] == 0:
                break
            step_ar += 1
        else:
            break
    
    # concat traj
    all_trajs, in_trajs, out_trajs = {}, {}, {}
    raw_trajs = {}
    try:
        for key in traj_dict.keys():
            all_trajs[key] = np.stack([d.numpy() for d in traj_dict[key]], axis=0)  # torch is slower
            in_trajs[key] = np.stack([d.numpy() for d in in_dict[key]], axis=0)
            out_trajs[key] = np.stack([d.numpy() for d in out_dict[key]], axis=0)
            raw_trajs[key] = np.stack([d.numpy() for d in raw_dict[key]], axis=0)
    except RuntimeError:
        raise NotImplementedError('fix to save traj information')
        for key in traj_dict.keys():
            all_trajs[key] = pad_and_stack(traj_dict[key], dim=0)
            in_trajs[key] = pad_and_stack(in_dict[key], dim=0)
            out_trajs[key] = pad_and_stack(out_dict[key], dim=0)
            raw_trajs[key] = pad_and_stack(raw_dict[key], dim=0)
    trajs = {'all': all_trajs, 'in': in_trajs, 'out': out_trajs, 'raw': raw_trajs}
    outputs['confidence_pos_traj'] = torch.stack(cfd_traj, dim=-1)
    return batch, outputs, trajs
    # return batch, outputs, {}

def pad_and_stack(tensor_list, dim):
    size_list = [tensor.size(0) for tensor in tensor_list]
    max_size = max(size_list)
    if tensor_list[0].dim() == 1:
        tensor_list = [F.pad(tensor, (0, max_size-size), mode='constant', value=-1)  # type = -1 means
                    for tensor, size in zip(tensor_list, size_list)]
    else:
        tensor_list = [F.pad(tensor, (0, 0, 0, max_size-size), mode='constant', value=0)
                    for tensor, size in zip(tensor_list, size_list)]
    return torch.stack(tensor_list, dim=dim)

def seperate_outputs2(batch, outputs, trajs, off_tqdm=False):
    
    # fetch the halfedge_index for each data
    num_graphs = batch.num_graphs
    try:
        data_list = batch.to_data_list()
    except RuntimeError:
        del batch['node_p2'], batch['halfedge_p2'], batch['halfedge_p1p2']
        data_list = batch.to_data_list()

    # # generated mols
    generated_list = []
    for i_mol in tqdm(range(num_graphs), desc='Seperating mols', total=num_graphs, disable=off_tqdm):
        # if 'pocket_center' in data_list[i_mol].keys:
        if 'pocket_center' in data_list[i_mol]:
            pocket_center = data_list[i_mol]['pocket_center'].cpu().numpy()
            if len(pocket_center) == 0:
                pocket_center = np.zeros([1, 3])
        else:
            pocket_center = np.zeros([1, 3])
        generated_list.append({
            'node': data_list[i_mol]['node_type'].cpu().numpy(),
            'pos': data_list[i_mol]['node_pos'].cpu().numpy(),
            'halfedge': data_list[i_mol]['halfedge_type'].cpu().numpy(),
            'halfedge_index': data_list[i_mol]['halfedge_index'].cpu().numpy(),
            'pocket_center': pocket_center,
        })
    
    # # model outputs
    # outputs_list = [] # TODO. not used yet
    node_sizes = torch.bincount(batch['node_type_batch']).tolist()
    halfedge_sizes = torch.bincount(batch['halfedge_type_batch']).tolist()
    outputs_dict = {}
    for key, value in outputs.items():
        if len(value) == len(batch['node_type_batch']):
            outputs_dict[key] = torch.split(value, node_sizes)
        elif len(value) == len(batch['halfedge_type_batch']):
            outputs_dict[key] = torch.split(value, halfedge_sizes)
        elif len(value) == len(batch):
            outputs_dict[key] = value
    # outputs_dict = {key:torch.split(value, (halfedge_sizes if 'edge' in key else node_sizes))
    #                 for key, value in outputs.items()}
    outputs_list = []
    for i_mol in range(num_graphs):
        output = {key:value[i_mol].cpu() for key, value in outputs_dict.items()}
        output.update({'halfedge_index': data_list[i_mol]['halfedge_index'].cpu()})
        output.update({'pocket_center': data_list[i_mol]['pocket_center'].cpu()})
        outputs_list.append(output)
    
    # # trajs
    if trajs is not None:
        node_split_size = batch['node_type_batch'].bincount().tolist()
        halfedge_split_size = batch['halfedge_type_batch'].bincount().tolist()
        traj_list_dict = {key:[] for key in trajs.keys()}
        for traj_who in trajs.keys():
            if isinstance(trajs[traj_who]['node'], torch.Tensor):
                node_split = torch.split(trajs[traj_who]['node'], node_split_size, dim=1)
                pos_split = torch.split(trajs[traj_who]['pos'], node_split_size, dim=1)
                halfedge_split = torch.split(trajs[traj_who]['halfedge'], halfedge_split_size, dim=1)
                traj_list_dict[traj_who] = [{
                    'node': node_split[i_mol].cpu().numpy(),
                    'pos': pos_split[i_mol].cpu().numpy(),
                    'halfedge': halfedge_split[i_mol].cpu().numpy(),
                } for i_mol in range(num_graphs)]
            else:
                indices_node = np.cumsum(node_split_size)[:-1]
                indices_halfedge = np.cumsum(halfedge_split_size)[:-1]
                node_split = np.split(trajs[traj_who]['node'], indices_node, axis=1)
                pos_split = np.split(trajs[traj_who]['pos'], indices_node, axis=1)
                halfedge_split = np.split(trajs[traj_who]['halfedge'], indices_halfedge, axis=1)
                traj_list_dict[traj_who] = [{
                    'node': node_split[i_mol],
                    'pos': pos_split[i_mol],
                    'halfedge': halfedge_split[i_mol],
                } for i_mol in range(num_graphs)]
    else:
        traj_list_dict = []
    
    return generated_list, outputs_list, traj_list_dict


def get_cfd_traj(cfd_pos_traj, atom_dim=0, steps=100):
    cfd_atoms = cfd_pos_traj.mean(dim=atom_dim)  # atom-wise mean
    if (len(cfd_atoms) != steps) and (len(cfd_atoms) % steps == 0):  # refine-based sampling
        cfd_atoms_reshape = cfd_atoms.view(-1, steps)
        std_cfd = cfd_atoms_reshape.std(dim=1)
        last_round = len(std_cfd[std_cfd > 0.01]) - 1
        cfd_atoms = cfd_atoms_reshape[last_round]
    cfd = cfd_atoms[-cfd_atoms.size(0)//2:].mean()
    if isinstance(cfd, torch.Tensor):
        cfd = cfd.item()
    return cfd

def post_process_generated(generated_list, outputs_list, traj_list_dict):
    mol_info_list = []
    for i_mol in range(len(generated_list)):
        mol_info = featurizer.decode_output(**generated_list[i_mol]) 
        mol_info.update(data_list[i_mol])  # add data info
        
        # reconstruct mols
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info)
            smiles = Chem.MolToSmiles(rdmol)
            if '.' in smiles:
                tag = 'incomp'
                pool.incomp.append(mol_info)
                logger.warning('Incomplete molecule: %s' % smiles)
            else:
                tag = ''
                pool.succ.append(mol_info)
                logger.info('Success: %s' % smiles)
        except MolReconsError:
            pool.bad.append(mol_info)
            logger.warning('Reconstruction error encountered.')
            smiles = ''
            # rdmol = Chem.MolFromSmiles(smiles)
            tag = 'bad'
            # raise NotImplementedError('fix to save information anyway')
            rdmol = create_sdf_string(mol_info)
        
        mol_info['rdmol'] = rdmol
        mol_info['smiles'] = smiles
        mol_info['tag'] = tag
        mol_info['output'] = outputs_list[i_mol]
        
        # get traj
        p_save_traj = np.random.rand()  # save traj
        if p_save_traj <  save_traj_prob:
            mol_traj = {}
            for traj_who in traj_list_dict.keys():
                traj_this_mol = traj_list_dict[traj_who][i_mol]
                for t in range(len(traj_this_mol['node'])):
                    mol_this = featurizer.decode_output(
                            node=traj_this_mol['node'][t],
                            pos=traj_this_mol['pos'][t],
                            halfedge=traj_this_mol['halfedge'][t],
                            halfedge_index=generated_list[i_mol]['halfedge_index'],
                            pocket_center=generated_list[i_mol]['pocket_center'],
                        )
                    mol_this = create_sdf_string(mol_this)
                    mol_traj.setdefault(traj_who, []).append(mol_this)
                    
            mol_info['traj'] = mol_traj
        mol_info_list.append(mol_info)


def get_mol_parts_linking(batch, device=None):
    parts = []
    for i_part in [1, 2]:
        # if f'gt_node_type_p{i_part}' not in batch.keys:
        if f'gt_node_type_p{i_part}' not in batch:
            continue
        part = {
            'gt_node_type': batch[f'gt_node_type_p{i_part}'].clone(),
            'gt_node_pos': batch[f'gt_node_pos_p{i_part}'].clone(),
            'gt_halfedge_type': batch[f'gt_halfedge_type_p{i_part}'].clone(),
        }
        parts.append(part)
    return parts
        

def seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge):
    outputs_pred = outputs

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_halfedge = (batch_halfedge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
        new_pred_this = [outputs_pred[0][ind_node],  # node type
                         outputs_pred[1][ind_node],  # node pos
                         outputs_pred[2][ind_halfedge]]  # halfedge type
                        
        halfedge_index_this = halfedge_index[:, ind_halfedge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        new_outputs.append({
            'node': new_pred_this[0],
            'pos': new_pred_this[1],
            'halfedge': new_pred_this[2],
            'halfedge_index': halfedge_index_this,
        })
    return new_outputs



def get_atom_and_bond(pred_node, pred_pos, pred_halfedge):
    """
    Get the atom and bond information from the prediction (latent space)
    pred_node: [n_nodes, n_node_types]
    pred_pos: [n_nodes, 3]
    pred_halfedge: [n_halfedges, n_edge_types]
    """
    # get atoms
    pred_atom = torch.softmax(pred_node, dim=-1)
    atom_prob, atom_type = torch.max(pred_atom, dim=-1)
    is_atom = (atom_types > 0)
    n_atoms = is_atom.sum().item()
    atom_types = atom_types[is_atom] - 1
    atom_prob = atom_prob_all[is_atom]
    atom_prob_dummy = atom_prob_all[~is_atom]
    atom_pos = pos_pred[is_atom]
    
    # get bonds for real atom 
    n_context = data.ligand_context_pos.size(0)
    n_compose = data.compose_pos.size(0)
    bond_index = data.bond_index
    bond_mask_tbp = data.bond_mask
    # pred_bond_prob = torch.softmax(pred_bond_logits, dim=-1)
    pred_bond_prob = pred_bond_logits
    bond_prob_tbp, bond_types_tbp = torch.max(pred_bond_prob, dim=-1)
    index_real_in_compose = torch.nonzero(is_atom) + n_compose
    # n_atoms_real = n_compose + n_atoms 
    # bond_mask_real = torch.stack([((bond[0]<n_atoms_real) and (bond[1]<n_atoms_real)) for bond in bond_index.T])
    bond_mask_real = torch.from_numpy(np.array([(
        ((bond[0] in index_real_in_compose) or (bond[0] < n_context)) and
        ((bond[1] in index_real_in_compose) or (bond[1] < n_context))
    ) for bond in bond_index.T]))
    
    bond_index_real = bond_index[:, (bond_mask_tbp & bond_mask_real)]
    bond_mask = bond_mask_real[bond_mask_tbp]
    bond_types_real = bond_types_tbp[bond_mask]
    bond_prob_real = bond_prob_tbp[bond_mask]

    is_positive_bond = bond_types_real > 0
    bond_types = bond_types_real[is_positive_bond]
    bond_prob = bond_prob_real[is_positive_bond]
    bond_index = bond_index_real[:, is_positive_bond]
    bond_prob_dummy = bond_prob_real[~is_positive_bond]

    n_protein = data.protein_pos.size(0)
    idx_changer = torch.zeros(len(is_atom), dtype=torch.long)
    idx_changer[is_atom] = torch.arange(n_atoms) + n_context
    idx_changer = torch.cat([
        torch.arange(n_context),
        torch.zeros(n_compose - n_context),
        idx_changer
    ], dim=-1)
    bond_index = idx_changer[bond_index]
    # bond_index = torch.where(bond_index<n_context, bond_index,
    #                         (bond_index-n_protein)[idx_changer])
    return {
        'atom': [atom_types, atom_prob, atom_prob_dummy, atom_pos],
        'bond': [bond_types, bond_prob, bond_prob_dummy, bond_index]
    }

    # ele_types = frag_atom_max[is_atom] - 1  # minus 1 for 0 = None atom
    # # ele_prob = frag_atom_prob[is_atom]
    # pos = frag_pos[is_atom]


def add_ligand_atom_to_data(data, element, pos, bond_types, bond_index, type_map=[6,7,8,9,15,16,17]):
    """
    """
    data = data.clone()
    n_atoms = len(element)

    data.ligand_context_pos = torch.cat([
        data.ligand_context_pos,
        pos.view(n_atoms, 3).to(data.ligand_context_pos)
    ], dim=0)

    data.ligand_context_feature_full = torch.cat([
        data.ligand_context_feature_full,
        F.one_hot(element, len(type_map)).to(data.ligand_context_feature_full), # (n_atoms, num_elements)
    ], dim=0)

    element = torch.LongTensor([type_map[e] for e in element])
    data.ligand_context_element = torch.cat([
        data.ligand_context_element,
        element.view(n_atoms).to(data.ligand_context_element)
    ])

    data.ligand_context_bond_index = torch.cat([
        data.ligand_context_bond_index,
        bond_index.to(data.ligand_context_bond_index),
        torch.stack([
            bond_index[1], bond_index[0]
        ], dim=0).to(data.ligand_context_bond_index)
    ], dim=-1)
    data.ligand_context_bond_type = torch.cat([
        data.ligand_context_bond_type,
        bond_types.to(data.ligand_context_bond_type),
        bond_types.to(data.ligand_context_bond_type)
    ], dim=-1)

    return data


def get_atom_and_bond(data, pred_atom_logits, pred_bond_logits, pos_pred):
    # get atoms
    # pred_atom_prob = torch.softmax(pred_atom_logits, dim=-1)  # has NOT passed softmax
    pred_atom_prob = pred_atom_logits  # has passed softmax
    atom_prob_all, atom_types = torch.max(pred_atom_prob, dim=-1)
    is_atom = (atom_types > 0)
    n_atoms = is_atom.sum().item()
    atom_types = atom_types[is_atom] - 1
    atom_prob = atom_prob_all[is_atom]
    atom_prob_dummy = atom_prob_all[~is_atom]
    atom_pos = pos_pred[is_atom]
    
    # get bonds for real atom 
    n_context = data.ligand_context_pos.size(0)
    n_compose = data.compose_pos.size(0)
    bond_index = data.bond_index
    bond_mask_tbp = data.bond_mask
    # pred_bond_prob = torch.softmax(pred_bond_logits, dim=-1)
    pred_bond_prob = pred_bond_logits
    bond_prob_tbp, bond_types_tbp = torch.max(pred_bond_prob, dim=-1)
    index_real_in_compose = torch.nonzero(is_atom) + n_compose
    # n_atoms_real = n_compose + n_atoms 
    # bond_mask_real = torch.stack([((bond[0]<n_atoms_real) and (bond[1]<n_atoms_real)) for bond in bond_index.T])
    bond_mask_real = torch.from_numpy(np.array([(
        ((bond[0] in index_real_in_compose) or (bond[0] < n_context)) and
        ((bond[1] in index_real_in_compose) or (bond[1] < n_context))
    ) for bond in bond_index.T]))
    
    bond_index_real = bond_index[:, (bond_mask_tbp & bond_mask_real)]
    bond_mask = bond_mask_real[bond_mask_tbp]
    bond_types_real = bond_types_tbp[bond_mask]
    bond_prob_real = bond_prob_tbp[bond_mask]

    is_positive_bond = bond_types_real > 0
    bond_types = bond_types_real[is_positive_bond]
    bond_prob = bond_prob_real[is_positive_bond]
    bond_index = bond_index_real[:, is_positive_bond]
    bond_prob_dummy = bond_prob_real[~is_positive_bond]

    n_protein = data.protein_pos.size(0)
    idx_changer = torch.zeros(len(is_atom), dtype=torch.long)
    idx_changer[is_atom] = torch.arange(n_atoms) + n_context
    idx_changer = torch.cat([
        torch.arange(n_context),
        torch.zeros(n_compose - n_context),
        idx_changer
    ], dim=-1)
    bond_index = idx_changer[bond_index]
    # bond_index = torch.where(bond_index<n_context, bond_index,
    #                         (bond_index-n_protein)[idx_changer])
    return {
        'atom': [atom_types, atom_prob, atom_prob_dummy, atom_pos],
        'bond': [bond_types, bond_prob, bond_prob_dummy, bond_index]
    }

    # ele_types = frag_atom_max[is_atom] - 1  # minus 1 for 0 = None atom
    # # ele_prob = frag_atom_prob[is_atom]
    # pos = frag_pos[is_atom]


def get_next_step(
        data_list,
        generated,
        # transform,
        type_map=[6,7,8,9,15,16,17],
        threshold=None,
    ):
    new_data_list = []
    n_mols = len(data_list)
    # is_finished = np.zeros(n_mols, dtype=bool)
    start_bond = 0
    for i in range(n_mols):
        parent_sample = deepcopy(data_list[i])
        
        # # has focal
        # has_focal = generated['has_focal'][i]
        # if not has_focal:
        #     is_finished[i] = True
        #     new_data_list.append(parent_sample)

        #     n_bond = parent_sample.bond_mask.int().sum()
        #     start_bond = start_bond + n_bond
        #     continue
        # else:  # get focal informaiton
        #     focal_prob = generated['focal_prob'][i].item()

        # # get things to add ( pos, atom, bond)
        frag_pos = generated['pred_pos'][i]  # (max_fragment, 3)
        frag_atom_pred = generated['pred_atom'][i]  # (max_fragment, num_elements)
        n_bond = parent_sample.bond_mask.int().sum()
        bond_types = generated['pred_bond'][start_bond:start_bond+n_bond]
        start_bond = start_bond + n_bond
        #TODO: add pos traj information
        
        # # get real atoms
        # frag_atom_prob = torch.softmax(frag_atom_pred, dim=-1)  # has not passed softmax
        # frag_atom_prob, frag_atom_max = torch.max(frag_atom_prob, dim=-1)
        # is_atom = (frag_atom_max > 0)
        # n_atoms = is_atom.sum().item()
        # ele_types = frag_atom_max[is_atom] - 1  # minus 1 for 0 = None atom
        # # ele_prob = frag_atom_prob[is_atom]
        # pos = frag_pos[is_atom]
        things = get_atom_and_bond(parent_sample, pred_atom_logits=frag_atom_pred,
                                  pred_bond_logits=bond_types, pos_pred=frag_pos)
        atom_types, atom_prob, atom_prob_dummy, atom_pos = things['atom']
        bond_types, bond_prob, bond_prob_dummy, bond_index = things['bond']
        
        # # add to ligand
        data_new = add_ligand_atom_to_data(
            parent_sample,
            element = atom_types,
            pos = atom_pos,
            bond_types = bond_types,
            bond_index = bond_index,
            type_map = type_map
        )

        # # log
        if hasattr(data_new, 'prob_atom'):
            data_new.prob_atom.append(atom_prob.cpu().detach().numpy())
            data_new.prob_atom.append(atom_prob_dummy.cpu().detach().numpy())
        else:
            data_new.prob_atom = [atom_prob.cpu().detach().numpy()]
            data_new.prob_atom.append(atom_prob_dummy.cpu().detach().numpy())
        if hasattr(data_new, 'prob_bond'):
            data_new.prob_bond.append(bond_prob.cpu().detach().numpy())
            data_new.prob_bond.append(bond_prob_dummy.cpu().detach().numpy())
        else:
            data_new.prob_bond = [bond_prob.cpu().detach().numpy()]
            data_new.prob_bond.append(bond_prob_dummy.cpu().detach().numpy())

        new_data_list.append(data_new)

    return new_data_list


def add_bond_to_data(
        data,
        bond_index,
        bond_type
):
    bond_index_all = torch.cat([bond_index, torch.stack([bond_index[1, :], bond_index[0, :]], dim=0)], dim=1)
    bond_type_all = torch.cat([bond_type, bond_type], dim=0)
    data.ligand_context_bond_index = bond_index_all
    data.ligand_context_bond_type = bond_type_all
    return data

def finish_step(data, edge_pred, edge_index):
    edge_pred = torch.softmax(edge_pred, dim=-1)
    edge_type = edge_pred.argmax(dim=-1)
    edge_prob = edge_pred[torch.arange(len(edge_type)), edge_type]
    # drop edge_type == 0
    is_bond = (edge_type > 0)
    bond_index = edge_index[:, is_bond]
    bond_type = edge_type[is_bond]
    bond_prob = edge_prob[is_bond]
    # add to data
    data = add_bond_to_data(data, bond_index, bond_type)
    data.prob_bond = bond_prob
    return data