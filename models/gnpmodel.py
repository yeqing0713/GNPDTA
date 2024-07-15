import torch, numpy
import torch.nn as nn
from torch_geometric.nn import SAGPooling, GCNConv, SAGEConv, global_mean_pool as gap, global_max_pool as gmp
import random

test = random.sample(range(0, 2), 2)


class TargetCNNNet(torch.nn.Module):
    def __init__(self,   n_filters = 32, dropout = 0.1):
        super(TargetCNNNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=[5, 5], stride=[3, 3], padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=[5, 5],  stride=[3, 3], padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters * 2)
        self.conv_3 = nn.Conv2d(in_channels=n_filters *2, out_channels=n_filters * 4, kernel_size=[3, 3],  stride=[2, 2], padding=1)
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, xt):
        conv_xt = self.conv_1(xt)
        conv_xt = self.bn1(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        conv_xt = self.conv_2(conv_xt)
        conv_xt = self.bn2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        conv_xt = self.conv_3(conv_xt)
        conv_xt = self.bn3(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        return conv_xt

class DrugCNNNet(torch.nn.Module):
    def __init__(self, n_filters = 32, dropout = 0.1):
        super(DrugCNNNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=[5, 5], stride=[3, 3], padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(in_channels=n_filters , out_channels=n_filters * 2, kernel_size=[5, 5],  stride=[3, 3], padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters * 2)
        self.conv_3 = nn.Conv2d(in_channels=n_filters *2, out_channels=n_filters * 4, kernel_size=[3, 3],  stride=[2, 2], padding=1)
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, xt):
        conv_xt = self.conv_1(xt)  # 512*(32*4)*128
        conv_xt = self.bn1(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        conv_xt = self.conv_2(conv_xt)  # 512*(32*2)*128
        conv_xt = self.bn2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        conv_xt = self.conv_3(conv_xt)  # 512*(32*1)*128
        conv_xt = self.bn3(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        return conv_xt

class FLDNN(torch.nn.Module):
    def __init__(self,input_dim, n_output, dropout):
        super(FLDNN, self).__init__()
        self.fc1= nn.Linear(input_dim, 512)
        self.bn1= nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, xc):
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.bn1(xc)

        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.bn2(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class gnp2dcnn(torch.nn.Module):
    def __init__(self, n_filters=16, num_features_xd=78, output_dim=1024, dropout=0.1, n_output=1):
        super(gnp2dcnn, self).__init__()
        self.DrugCNNNet = DrugCNNNet()
        self.TargetCNNNet = TargetCNNNet()

        self.FLDNN_T = FLDNN( 2304, n_output, dropout)

    def forward(self, data):
        target = data.target  # 512x1000
        drug = data.drug
        num = len(drug)
        drug = torch.from_numpy(numpy.array(drug))
        target = torch.from_numpy(numpy.array(target))

        embedded_xd = drug.reshape([num, 60, 60]).to('cuda:0')
        embedded_xt = target.reshape([num, 60, 60]).to('cuda:0')

        embedded_xd = embedded_xd.unsqueeze(1)
        embedded_xt = embedded_xt.unsqueeze(1)
        #with torch.no_grad():
        embedded_xd = self.DrugCNNNet(embedded_xd)
        embedded_xt = self.TargetCNNNet(embedded_xt)
        #print(embedded_xt.shape)
        #print(embedded_xd.shape)
        embedded_xd = embedded_xd.view(-1, 1152)
        embedded_xt = embedded_xt.view(-1, 1152)

        outs = []
        conv_xd_xt = torch.cat(( embedded_xd, embedded_xt), 1)
        #print(conv_xd_xt.shape)
        conv_xd_xt_T, out_T = self.FLDNN_T(conv_xd_xt)
        outs.append(out_T)

        return outs

def generate_inxs_drug(total_num, left_num, subspace_num):
    start_inxss = []
    end_inxss = []
    step_shift = ((total_num - left_num) / float(subspace_num))
    for jj in range(subspace_num):
        start_inx = int(jj * step_shift)
        end_inx = start_inx + left_num
        if end_inx > total_num:
            end_inx = total_num
            start_inx = end_inx - left_num
            if (start_inx < 0):
                start_inx = 0
        start_inxss.append(start_inx)
        end_inxss.append(end_inx)
    return start_inxss, end_inxss


import lightning.pytorch as pl
from torch import optim, nn, utils, Tensor
class LitDTA(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        outs = self.model(batch)
        total_loss = 0
        for ii in range(len(outs)):
            out = outs[ii]
            loss = self.loss_fn(out, batch.y.view(-1, 1).float())
            if ii == 0 or ii == 1:
                total_loss = total_loss + loss
            else:
                total_loss = total_loss + loss/len(outs)/2
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        outs = self.model(batch)
        loss = self.loss_fn(outs[0], batch.y.view(-1, 1).float())
        self.log("val_loss", loss)
        return loss

    def forward(self, batch):
        return self.model(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.learning_rate)
        #return bnb.optim.Adam8bit(self.parameters(), lr=0.001, betas=(0.9, 0.995))
        return optimizer