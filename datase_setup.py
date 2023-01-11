import os.path as osp
from tqdm import  tqdm
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Dataset, download_url, Data, DataLoader
import pandas as pd
import os
import numpy as np
import requests
import csv

def visualize_molecule():
    data=pd.read_csv('data/raw/HIV.csv')
    molecule=[]
    for i in range(16):
        # print(data['smiles'][i])
        mol_obj=Chem.MolFromSmiles(data['smiles'][i])
        str_ = 'images/molecule' + str(i) + '.png'
        Draw.MolToFile(mol_obj, str_)

        # str_ = 'images/molecule'+str(i)+'.png'


class MyOwnDataset(Dataset):
    def __init__(self, root, file_name_, transform=None, pre_transform=None, pre_filter=None):
        self.file_name_ = file_name_
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [self.file_name_]

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0])
        index= len(data)
        return [f'molecule_{name}.pt' for name in range(index)]

    def download(self):
        temp_file_name=os.path.join(self.raw_dir,'HIV.csv')
        url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
        download = requests.get(url)
        with open(temp_file_name, 'w') as f:
            for line in download.iter_lines():
                    f.write(line.decode('utf-8')  + '\n')
    def process(self):
        data = pd.read_csv(self.raw_paths[0])
        for i, mol in tqdm(data.iterrows(), total=data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol['smiles'])
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol_obj)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(mol["HIV_active"])

            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol["smiles"]
                        )

            torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'molecule_{i}.pt'))

    def _get_node_features(self, mol):
            all_node_feats = []

            for atom in mol.GetAtoms():
                node_feats = []
                # Feature 1: Atomic number
                node_feats.append(atom.GetAtomicNum())
                # Feature 2: Atom degree
                node_feats.append(atom.GetDegree())
                # Feature 3: Formal charge
                node_feats.append(atom.GetFormalCharge())
                # Feature 4: Hybridization
                node_feats.append(atom.GetHybridization())
                # Feature 5: Aromaticity
                node_feats.append(atom.GetIsAromatic())
                # Feature 6: Total Num Hs
                node_feats.append(atom.GetTotalNumHs())
                # Feature 7: Radical Electrons
                node_feats.append(atom.GetNumRadicalElectrons())
                # Feature 8: In Ring
                node_feats.append(atom.IsInRing())
                # Feature 9: Chirality
                node_feats.append(atom.GetChiralTag())

                # Append node features to matrix
                all_node_feats.append(node_feats)

            all_node_feats = np.asarray(all_node_feats)
            return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
            """
            This will return a matrix / 2d array of the shape
            [Number of edges, Edge Feature size]
            """
            all_edge_feats = []

            for bond in mol.GetBonds():
                edge_feats = []
                # Feature 1: Bond type (as double)
                edge_feats.append(bond.GetBondTypeAsDouble())
                # Feature 2: Rings
                edge_feats.append(bond.IsInRing())
                # Append node features to matrix (twice, per direction)
                all_edge_feats += [edge_feats, edge_feats]

            all_edge_feats = np.asarray(all_edge_feats)
            return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
            """
            We could also use rdmolops.GetAdjacencyMatrix(mol)
            but we want to be sure that the order of the indices
            matches the order of the edge features
            """
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices += [[i, j], [j, i]]

            edge_indices = torch.tensor(edge_indices)
            edge_indices = edge_indices.t().to(torch.long).view(2, -1)
            return edge_indices

    def _get_labels(self, label):
            label = np.asarray([label])
            return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
            """ - Equivalent to __getitem__ in pytorch
                - Is not needed for PyG's InMemoryDataset
            """
            # if self.test:
            #     data = torch.load(os.path.join(self.processed_dir,
            #                                    f'data_test_{idx}.pt'))
            # else:
            data = torch.load(os.path.join(self.processed_dir,
                                               f'molecule_{idx}.pt'))
            return data
if __name__=='__main__':

    S = MyOwnDataset(root='data',file_name_='HIV.csv')
    print(S[0].edge_attr)
    load = DataLoader(S, batch_size=2, shuffle=True)
    for _, item in enumerate(S):
        print(item)
#     visualize_molecule()