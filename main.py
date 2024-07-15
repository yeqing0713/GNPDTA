# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from torch import optim, nn, utils, Tensor
import torch
import os
from utils_my import *
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
import lightning.pytorch as pl
from models.gnpmodel import LitDTA, gnp2dcnn
from lightning.pytorch.tuner import Tuner

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            output = outputs[0]
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def train(model, device, train_loader, optimizer, epoch):
    #print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        outs = model(data)
        total_loss = 0
        for ii in range(len(outs)):
            out = outs[ii]
            loss = loss_fn(out, data.y.view(-1, 1).float().to(device))
            if ii == 0:
                total_loss = total_loss + loss
            else:
                total_loss = total_loss + loss/len(outs)/2

        total_loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           total_loss.item()))


from tqdm import tqdm
from utils_my import *
from lifelines.utils import concordance_index

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from scipy import stats
import csv
def train_and_test(dataset, f_out):
    cuda_name = "cuda:0"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model_st = 'gnp2dcnn'
    print('\nrunning on ', model_st + '_' + dataset )
    print('\nrunning on ', model_st + '_' + dataset, file=f_out)
    # print('\nrunning on ', model_st + '_' + dataset, file=fout)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
        print('please run create_data.py to prepare data in pytorch format!', file=f_out)
    else:

        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')
        print(train_data)

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        DTA = LitDTA(gnp2dcnn())
        DTA.to(device)
        total_params = sum(p.numel() for p in DTA.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in DTA.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        unupdate_epoch = 0
        for epoch in tqdm(range(NUM_EPOCHS)):
            unupdate_epoch = unupdate_epoch + 1
            #trainer = pl.Trainer(max_epochs=1, precision='16', benchmark=True)
            #trainer.fit(model=DTA, train_dataloaders=train_loader)
            train(DTA, device, train_loader, DTA.configure_optimizers(), epoch + 1)
            if epoch < 300:
                #print(epoch)
                continue
            G, P = predicting(DTA, device, test_loader)
            ret = [mse(G, P), concordance_index(G, P), mse(G, P, squared=False), r2s(G, P), stats.spearmanr(G, P)[0]]

            print('epoch ', epoch, '; mse,ci,rmse,r2, spear:',
                  ret[0], ret[1], ret[2], ret[3], ret[4], model_st, dataset)
            print('epoch ', epoch, '; mse,ci,rmse,r2, spear:',
                  ret[0], ret[1], ret[2], ret[3], ret[4], model_st, dataset, file=f_out)


if __name__ == '__main__':
    f_out = open("2DT_GCN_new_CNN_new_MS_4812_XX.txt", "a")
    torch.set_float32_matmul_precision('medium')
    TRAIN_BATCH_SIZE = 1280
    TEST_BATCH_SIZE = 1280
    LR = 0.001  # 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 3000
    datasets = ['davis', 'kiba']
    loss_fn = nn.MSELoss()
    for dataset in datasets:
        train_and_test(dataset, f_out)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
