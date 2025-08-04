import os
from tqdm import tqdm
import pickle
import argparse
from glob import glob
from multiprocessing import Pool
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

from feature_utils import get_ligand_info, get_protein_info, get_chem_feats, read_mol, get_coords

import warnings
import random

warnings.filterwarnings('ignore')




def process_single(items):
    input_path, output_path = items
    new_data = {}
    name = os.path.basename(input_path)


    try:
        mol_file = glob(f'{input_path}/ligand.*')[0]
    except IndexError:
        print(f'{name}: 未找到配体文件')
        return False

    pro_file = f'{input_path}/receptor.pdb'
    pocket_file = f'{input_path}/pocket.txt'
    complex_file = f'{input_path}/complex.pdb'


    try:
        with open(pocket_file, 'r') as f:
            poc_res = f.read().splitlines()
    except FileNotFoundError:
        print(f'{name}: 未找到口袋文件')
        return False


    input_mol = read_mol(mol_file)
    try:
        mol, smiles, coordinate_list = get_coords(input_mol)
    except Exception as e:
        print(f'{name}: 生成配体坐标失败: {e}')
        return False

    try:
        holo_coordinates = extract_holo_coordinates(complex_file, name)
        new_data['holo_coordinates'] = holo_coordinates
    except Exception as e:
        print(f'{name}: 提取 holo_coordinates 失败: {e}')
        return False

    try:
        lig_atoms, lig_atom_feats, lig_edges, lig_bonds = get_ligand_info(mol)
        poc_pos, poc_atoms, poc_atom_feats, poc_edges, poc_bonds = get_protein_info(pro_file, poc_res)
    except Exception as e:
        print(f'{name}: 提取配体/蛋白质信息失败: {e}')
        return False


    new_data.update({
        'atoms': lig_atoms,
        'coordinates': coordinate_list,
        'pocket_atoms': poc_atoms,
        'pocket_coordinates': poc_pos,
        'smi': smiles,
        'pocket': name,
        'lig_feats': lig_atom_feats,
        'lig_bonds': lig_edges,
        'lig_bonds_feats': lig_bonds,
        'poc_feats': poc_atom_feats,
        'poc_bonds': poc_edges,
        'poc_bonds_feats': poc_bonds,
        'mol': mol
    })


    try:
        new_data = get_chem_feats(new_data)
    except Exception as e:
        print(f'{name}: 计算化学特征失败: {e}')
        return False


    try:
        with open(output_path, 'wb') as f_out:
            pickle.dump(new_data, f_out)
    except Exception as e:
        print(f'{name}: 保存处理数据失败: {e}')
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='预处理数据集')
    parser.add_argument('--input_path', type=str, default='pdb', help='')
    parser.add_argument('--output_path', type=str, default='pkl', help='')
    parser.add_argument('--threads', type=int, default=8, help='')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    input_data_list = os.listdir(args.input_path)
    output_data_list = [os.path.join(args.output_path, f'{x}.pkl') for x in input_data_list]
    input_data_list = [os.path.join(args.input_path, x) for x in input_data_list]
    data_list = list(zip(input_data_list, output_data_list))

    with Pool(args.threads) as pool:
        for success in tqdm(pool.imap(process_single, data_list), total=len(data_list)):
            if not success:
                print("处理某个项目失败。")


if __name__ == '__main__':
    main()
