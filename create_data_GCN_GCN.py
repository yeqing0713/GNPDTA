import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils_two_tranformer import *
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import transformers
from tokenizers import ByteLevelBPETokenizer
pd.read_csv('data/davis_train.csv')
print("OK!")
from torchdrug import data, utils
from torchdrug.core import Registry as R

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]

    return x



def seq_cat_transformer_target_GCN(prot, tokenizer, model, curr_i, total_i):
    m_n = 64
    bb = prot
    target_len = len(bb)

    feas = seq_feas[target_features[prot]]
    #print(feas)
    #print('prot', curr_i, total_i)
    return feas

def seq_cat_transformer_drug_GCN(prot, tokenizer, model, curr_i, total_i):
    m_n = 64
    bb = prot
    target_len = len(bb)

    feas = drug_GCN_feas[drug_GCN_features[prot]]
    print(feas.shape)
    print('prot', curr_i, total_i)
    return feas



def seq_cat_transformer_target_GCN_global(prot, tokenizer, model, curr_i, total_i):
    m_n = 64
    bb = prot
    target_len = len(bb)
    #print(target_len)
    step_shift = ((target_len - 128) / (m_n - 1))
    feas = torch.zeros(m_n, 60)
    for jj in range(m_n):
        start_inx = int(jj * step_shift)
        end_inx = start_inx + 128
        if end_inx > target_len:
            end_inx = target_len
            start_inx = end_inx - 128
            if (start_inx < 0):
                start_inx = 0

        text = bb[start_inx:end_inx]
        try:
            graph1 = data.Protein.from_sequence(text, atom_feature='default',  bond_feature='default')
            with torch.no_grad():
                output = target_model(graph1, graph1.node_feature.float())
                output = output["graph_feature"]
            #print(output["graph_feature"])
        #print(type(output["graph_feature"]))
        #print(output.shape)
            feas[jj, :] = output[0, :]
        except:
            print(text)

    #print(feas.shape)
    feas = feas.reshape((1, feas.shape[0] * feas.shape[1]))
    feas = feas.squeeze(0)
    #x = np.zeros(feas.shape)
    #for ii in range(feas.shape[0]):
     #   x[ii] = feas[ii]
    #print(x.shape)
    #print(feas.shape)
    print('prot', curr_i, total_i)
    return feas.numpy()

def seq_cat_transformer_drug_GCN_global(prot, tokenizer, model, curr_i, total_i):
    m_n = 64
    bb = prot
    feas = torch.zeros(m_n, 60)
    graph1 = data.Molecule.from_smiles(bb, atom_feature='default', bond_feature='default')
    with torch.no_grad():
        output = drug_model(graph1, graph1.node_feature.float())
        output = output["node_feature"]
    print(output.shape)
    [r, c] = output.shape
    if r > m_n:
        feas[0 : m_n, :] = output[0 : m_n, :]
    else:
        feas[0 : r, :] = output[0 : r, :]

    print('prot', curr_i, total_i)
    #print(feas.shape)
    return feas.numpy()


    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# create graph for all SMILES
compound_iso_smiles = []
for dt_name in ['davis', 'kiba', 'DTC', 'Metz', 'ToxCast']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


# create graph for all SMILES

from torchdrug import core, datasets, tasks, models
#datasets = ['davis', 'kiba', 'DTC', 'Metz', 'ToxCast']
datasets = ['davis', 'kiba', 'DTC', 'Metz']
# convert to PyTorch data format


drug_model = models.GIN(input_dim=67,
                       hidden_dims=[300, 300, 300, 300, 60],
                       edge_input_dim=18,
                       batch_norm=True, readout="mean")
state_dict = torch.load("drug_gin.model")
drug_model.load_state_dict(state_dict, strict=True)


target_model = models.GIN(input_dim=67,
                       hidden_dims=[300, 300, 300, 300, 60],
                       edge_input_dim=18,
                       batch_norm=True, readout="mean")
state_dict = torch.load("target_gin.model")
target_model.load_state_dict(state_dict, strict=True)


#with open('data.pkl', 'rb') as f:
 #   loaded_data = pickle.load(f)

compound_iso_smi = []
for dt_name in ['davis', 'kiba', 'DTC', 'Metz']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smi += list(df['compound_iso_smiles'])
compound_iso_smi = set(compound_iso_smi)
drug_GCN_features = {}
drug_GCN_feas = []
num = 0
print(len(compound_iso_smi))
for seq in compound_iso_smi:
    #print(seq)
    fea = seq_cat_transformer_drug_GCN_global(seq, tokenizer1, model1, num, len(compound_iso_smi))
    print(fea.shape)
    drug_GCN_feas.append(fea)
    drug_GCN_features[seq] = num
    num = num + 1
with open('drug_GCN_features.pkl', 'wb') as f:
    pickle.dump(drug_GCN_features, f)
with open('drug_GCN_feas.pkl', 'wb') as f:
    pickle.dump(drug_GCN_feas, f)


#pic2 = open(r'drug_GCN_features.pkl','rb')
#drug_GCN_features = pickle.load(pic2)
#pic2 = open(r'drug_GCN_feas.pkl','rb')
#drug_GCN_feas = pickle.load(pic2)

compound_iso_seq = []
for dt_name in ['davis', 'kiba', 'DTC', 'Metz']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_seq += list(df['target_sequence'])
compound_iso_seq = set(compound_iso_seq)
target_features = {}
seq_feas = []
num = 0
for seq in compound_iso_seq:
    fea = seq_cat_transformer_target_GCN_global(seq, [], [], num, len(compound_iso_seq))
    seq_feas.append(fea)
    target_features[seq] = num
    num = num + 1
with open('target_features.pkl', 'wb') as f:
    pickle.dump(target_features, f)
with open('seq_feas.pkl', 'wb') as f:
    pickle.dump(seq_feas, f)

pic2 = open(r'target_features.pkl','rb')
target_features = pickle.load(pic2)
pic2 = open(r'seq_feas.pkl','rb')
seq_feas = pickle.load(pic2)


for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):

        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
        #XT = [seq_cat(t) for t in test_prots]
        #XT = [t for t in test_prots]

        XD = [seq_cat_transformer_drug_GCN(test_drugs[d], tokenizer1, model1, d, len(test_drugs)) for d in range(len(test_drugs))]
        XT = [seq_cat_transformer_target_GCN(test_prots[t], tokenizer, model, t, len(test_prots) ) for t in range(len(test_prots))]

        test_prots = [seq_cat(t) for t in test_prots]
        test_drugs, test_drugs_trans, test_prots, test_target_trans, test_Y = np.asarray(test_drugs), np.asarray(
            XD), np.asarray(test_prots), np.asarray(XT), np.asarray(test_Y)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        #test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xd_trans = test_drugs_trans,xt=test_prots, y=test_Y, smile_graph=smile_graph)
        test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xd_trans=test_drugs_trans, xt=test_prots, xt_trans=test_target_trans, y=test_Y, smile_graph=smile_graph)
        test_drugs = []
        test_drugs_trans = []
        test_prots = []
        test_Y = []

        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        print(train_drugs[0])
        #XT = [seq_cat(t) for t in train_prots]
        #XT = [t for t in train_prots]
        XD = [seq_cat_transformer_drug_GCN(train_drugs[d], tokenizer1, model1, d, len(train_drugs)) for d in
              range(len(train_drugs))]
        XT = [seq_cat_transformer_target_GCN(train_prots[t], tokenizer, model, t, len(train_prots)) for t in range(len(train_prots))]
        train_prots = [seq_cat(t) for t in train_prots]
        train_drugs, train_drugs_trans, train_prots, train_target_trans, train_Y = np.asarray(train_drugs), np.asarray(XD), np.asarray(train_prots), np.asarray(XT), np.asarray(train_Y)
        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xd_trans=train_drugs_trans, xt=train_prots, xt_trans=train_target_trans, y=train_Y, smile_graph=smile_graph)

        #print('preparing ', dataset + '_train.pt in pytorch format!')
        #train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xd_trans=train_drugs_trans, xt=train_prots, y=train_Y, smile_graph=smile_graph)

        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        
