import torch
import sys
import torch_geometric as tg
import pandas as pd
import csv
import subprocess

from ase import Atoms, Atom
from ase.neighborlist import neighbor_list

import numpy as np  

from pathlib import Path
from tqdm import tqdm

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

sys.path.append(getGitRoot())

def load_data(filename):
    df = pd.read_json(filename)
    df['structure'] = df.atoms.apply(lambda x: Atoms(**x))
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))
    
    return df, species

# build data
def build_data(entry, type_encoding, type_onehot, r_max):
    # print(entry)
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
        # hform=torch.from_numpy(np.array([entry.hform])).unsqueeze(0),
        id=entry.id
    )
    
    return data

base_dir = getGitRoot()

model_name = 'model_250123_full_model.torch'

"Hyperparameter"
r_max = 5 # cutoff radius, MUST ENSURE SAME AS TRAINING SCRIPT

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('torch device:' , device)
model = torch.load(model_name, map_location=device)

# one-hot encoding atom type and mass
type_encoding = {}
specie_am = []
for Z in tqdm(range(1, 119), bar_format=bar_format):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))

model.pool = True

df1, species = load_data(Path(base_dir) / 'datasets' / 'test.json')

df1['data'] = df1.progress_apply(lambda x: build_data(x, type_encoding, type_onehot, r_max), axis=1)
dataloader = tg.loader.DataLoader(df1['data'].values, batch_size=1)

model.to(device)
model.eval()
losses = []

with open('hform_predictions.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["id", "hform"])

with torch.no_grad():
    with open('predictions.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        i0 = 0
        for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
            d.to(device)
            output = model(d)
            # actual_hform = float(d.hform.cpu().numpy())
            hform = float(output.item()) 
            id = int(d.id.item())
            csv_writer.writerow([id, hform])
