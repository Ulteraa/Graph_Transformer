import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from rdkit import Chem
import pandas as pd
import torch.optim as optim
import torch_geometric.nn as GNN
from datase_setup import  MyOwnDataset
from torch_geometric.data import DataLoader
import csv
import seaborn as sns
import sklearn.metrics as sk_metric
from tqdm import tqdm
def balanced_data():
    file = "HIV.csv"
    data_ = "data_file.csv"
    negative_ = []; posetive_ = []
    with open(file, 'r') as origFile:
            csvreader = csv.DictReader(origFile)
            for line in csvreader:
                if line['HIV_active'] == '0':
                    negative_.append(line)
                else:
                    posetive_.append(line)
    header=['smiles', 'activity', 'HIV_active']
    with open(data_, 'w', newline='') as prosses_:
        csvwriter = csv.writer(prosses_)
        csvwriter.writerow(header)
        for index in range(len(posetive_)):
            csvwriter.writerow([posetive_[index]['smiles'], posetive_[index]['activity'], int(posetive_[index]['HIV_active'])])
            csvwriter.writerow([negative_[index]['smiles'], negative_[index]['activity'], int(negative_[index]['HIV_active'])])
    prosses_.close()
    origFile.close()

    # data = pd.read_csv('HIV.csv')
    # negative_ = []; posetive = []
    # for i in range(len(data)):
    #     negative_.append(i) if data.values[i][2]==0 else posetive.append(i)

class Graph_Transformer(nn.Module):
    def __init__(self, feature_dim, edge_dim):
        super(Graph_Transformer, self).__init__()
        self.embedding_size = 64
        self.n_heads = 3
        self.n_class = 2
        self.n_layers = 4
        self.dropout_rate = 0.2
        self.top_k_ratio = 0.5
        self.top_k_every_n = 1
        self.dense = 256
        self.edge_dim = edge_dim
        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

       #----------------------------------------------------------------------
        self.conv1 = GNN.TransformerConv(feature_dim,
                                     self.embedding_size,
                                     heads=self.n_heads,
                                     dropout=self.dropout_rate,
                                     edge_dim=self.edge_dim,
                                     beta=True)

        self.transf1 = nn.Linear(self.embedding_size * self.n_heads, self.embedding_size)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)

       #----------------------------------------------------------------
        for i in range(self.n_layers):
            self.conv_layers.append(GNN.TransformerConv(self.embedding_size,
                                                    self.embedding_size,
                                                    heads = self.n_heads,
                                                    dropout = self.dropout_rate,
                                                    edge_dim = self.edge_dim,
                                                    beta=True))

            self.transf_layers.append(nn.Linear(self.embedding_size * self.n_heads, self.embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(self.embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(GNN.TopKPooling(self.embedding_size, ratio=self.top_k_ratio))

        #----------------------------------------------------------------
        self.drop_out = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.embedding_size * 2, self.dense)
        self.linear2 = nn.Linear(self.dense, int(self.dense / 2))
        self.linear3 = nn.Linear(int(self.dense / 2), self.n_class)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        #--------------------------------------------------------------------
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                # Add current representation
                global_representation.append(torch.cat([GNN.global_max_pool(x, batch_index), GNN.global_mean_pool(x, batch_index)], dim=1))
#----------------------------------------------------------------------------------------------
        x = sum(global_representation)

        # Output block
        x = self.relu(self.linear1(x))
        x = self.drop_out(x)
        x = self.relu(self.linear2(x))
        x = self.drop_out(x)
        x = self.linear3(x)

        return x


def train(GGN_model, optimizer, train_data, criterion, device, epoch):
    loss = 0
    for index, batch in enumerate(tqdm(train_data)):
        batch = batch.to(device)
        predict_ = GGN_model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
        loss_ = criterion(predict_, batch.y)
        loss += loss_
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
    return loss
    # if epoch % 100 == 0:
    #     print(f'loss at epoch {epoch} is equal to {loss}')

def train_fn():
    train_dataset = MyOwnDataset(root='prosses_data/train', file_name_='train.csv')
    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
    GNN_model = Graph_Transformer(feature_dim=train_dataset[0].x.shape[1], edge_dim=train_dataset[0].edge_attr.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GNN_model = GNN_model.to(device)
    optimizer = optim.SGD(GNN_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # scheduler=optim.lr_scheduler.ExponentialLR()
    loss_arr = []
    for epoch in range(2001):
        loss = train(GNN_model, optimizer, train_data, criterion, device, epoch)
        loss_arr.append(loss_arr)
        if epoch % 100 == 0:
            print(f'testing the method with various metrics at epoch {epoch}')
            with torch.no_grad():
                 test(GNN_model, device)
    visulize(loss_arr)

def test(GNN_model, device):
    test_dataset = MyOwnDataset(root='prosses_data/test', file_name_='test.csv')
    print(len(test_dataset))
    batch = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _, batch in enumerate(batch):
            batch = batch.to(device)
            predict_ = GNN_model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
            predict_ = torch.argmax(predict_, dim=1)
            compute_acc(predict_.cpu().detach().numpy(), batch.y.cpu().detach().numpy())

def visulize(loss_arr):
    losses_float = [float(loss_arr.cpu().detach().numpy()) for loss in loss_arr]
    plt_ = sns.lineplot(losses_float)
    plt_.set(xlabel='epoch', ylabel='error')
    plt.savefig('train.png')

def compute_acc(predic, ground_trouth):
    print(f"\n Confusion matrix: \n {sk_metric.confusion_matrix(predic, ground_trouth)}")
    print(f"F1 Score: {sk_metric.f1_score(ground_trouth, predic)}")
    print(f"Accuracy: {sk_metric.accuracy_score(ground_trouth, predic)}")
    prec = sk_metric.precision_score(ground_trouth, predic)
    rec = sk_metric.recall_score(ground_trouth, predic)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    roc = sk_metric.roc_auc_score(ground_trouth, predic)
    print(f"ROC AUC: {roc}")

if __name__=='__main__':
    #balanced_data()
    train_fn()













