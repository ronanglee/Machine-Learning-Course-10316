import torch
import sys
import os
import torch_geometric as tg
import pandas as pd
from pathlib import Path

import subprocess
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list

import numpy as np  

import time
from tqdm import tqdm

def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

sys.path.append(getGitRoot())

from sklearn.model_selection import train_test_split
from gnn_e3nn.e3nn_scripts.utils.utils_model import PeriodicNetwork, train

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


def load_data(filename):
    df = pd.read_json(filename)
    df['structure'] = df.atoms.apply(lambda x: Atoms(**x))
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))
    df['hform'] = df.hform.apply(np.array)

    return df, species

def build_data(entry, type_encoding, type_onehot, r_max):
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        form=entry.formula,  # chemical formula
        x=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        hform=torch.from_numpy(np.array([entry.hform])).unsqueeze(0)
    )
    
    return data

def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

cwd = Path(__file__).parent.stem
parent = Path(Path(__file__)).parent.parent.name
run_name = f'model_{parent}_{cwd}'
full_cwd = Path(Path(__file__).parent)

"Hyperparameters"
r_max = 5 # cutoff radius
batch_size = 15
em_dim = 32
max_iter = 50
learning_rate = 0.005
l_max = 2

base_dir = getGitRoot()

df, species = load_data(Path(base_dir) / 'datasets' / 'train.json')

# one-hot encoding atom type and mass
type_encoding = {}
specie_am = []
for Z in tqdm(range(1, 119), bar_format=bar_format):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))

df['data'] = df.progress_apply(lambda x: build_data(x, type_encoding, type_onehot, r_max), axis=1)

idx_train, idx_valid = train_test_split(df.index, test_size=0.01, random_state=40)

print('number of training examples:', len(idx_train))
print('number of validation examples:', len(idx_valid))
n_train = get_neighbors(df, idx_train)
n_valid = get_neighbors(df, idx_valid)

dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)

out_dim = 1
model = PeriodicNetwork(
    in_dim=118,                            # dimension of one-hot encoding of atom type
    em_dim=em_dim,                         # dimension of atom-type embedding
    irreps_in=str(em_dim)+"x0e",           # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    irreps_out=str(out_dim)+"x0e",         # out_dim scalars (L=0 and even parity) to output
    irreps_node_attr=str(em_dim)+"x0e",    # em_dim scalars (L=0 and even parity) on each atom to represent atom type
    layers=2,                              # number of nonlinearities (number of convolutions = layers + 1)
    mul=32,                                # multiplicity of irreducible representations
    lmax=l_max,                                # maximum order of spherical harmonics
    max_radius=r_max,                      # cutoff radius for convolution
    num_neighbors=n_train.mean(),          # scaling factor based on the typical number of neighbors
    reduce_output=True                     # whether or not to aggregate features of all atoms at the end
)

opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)

model.pool = True

train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name, full_cwd,
      max_iter=max_iter, scheduler=scheduler, device=device)


