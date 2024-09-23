import os
import copy
import numpy as np
from tqdm import tqdm
import torch
import pickle


import sys

from torch.optim.lr_scheduler import StepLR

sys.path.append('../')

from model import ANN
from dataprocessing import nn_seq_wind, save_object, read_object
from optimizer import AlgoOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_val_loss(model, Val):
    model.eval()
    loss_function = torch.nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)

def train(ann, main):
    ann.train()
    Dtr, Val, Dte = nn_seq_wind(ann.name, ann.B)
    ann.len = len(Dtr)
    
    print("-------------------------------Training the Data-------------------------------")
       
    loss_function = torch.nn.MSELoss().to(device)
    x = copy.deepcopy(ann)
    optimizer = AlgoOptimizer(ann.parameters(), lr=ann.lr, weight_decay=1e-5)
    lr_step = StepLR(optimizer, step_size=20, gamma=0.0001)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(ann.K)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = ann.forward(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(main.control, ann.control)
        lr_step.step()
        # validation
        val_loss = get_val_loss(ann, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(ann)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        ann.train()

    ann = copy.deepcopy(best_model)
    temp = {}
    for k, v in ann.named_parameters():
        temp[k] = v.data.clone()

    for k, v in x.named_parameters():
        local_steps = ann.K * len(Dtr)
        ann.control[k] = ann.control[k] - main.control[k] + (v.data - temp[k]) / (local_steps * ann.lr)
        ann.delta_y[k] = temp[k] - v.data
        ann.delta_control[k] = ann.control[k] - x.control[k]

    return ann

def aggregation(N, main):
    delta_x = {}
    delta_c = {}
    
    for k, v in main.named_parameters():
        delta_x[k] = torch.zeros_like(v.data)
        delta_c[k] = torch.zeros_like(v.data)

    for i in range(N):
        client = read_object("clients/client"+str(i)+".pkl")
        for k, v in client.named_parameters():
            delta_x[k] += client.delta_y[k] / N  # averaging
            delta_c[k] += client.delta_control[k] / N  # averaging

    Ng = 1
    for k, v in main.named_parameters():
        v.data += (Ng)*delta_x[k].data
        main.control[k].data += delta_c[k].data * (N / N)

    return main

def caller(client_no, round_no):
    params = read_object("defaults.pkl")
    
    nn = ANN(input_dim = params["input_dim"], name = params["clients"][client_no], B = params["B"], K = params["K"], lr = params["lr"]).to(device)

    for k, v in nn.named_parameters():
        nn.control[k] = torch.zeros_like(v.data)
        nn.delta_control[k] = torch.zeros_like(v.data)
        nn.delta_y[k] = torch.zeros_like(v.data)

    if round_no == 0:
        main = read_object("main.pkl")
    else:
        main = read_object("clients/client"+str(client_no)+"_main.pkl")
        main = aggregation(params["N"], main)

    save_object(main, "clients/client"+str(client_no)+"_main.pkl")

    nn = train(nn, main)

    save_object(nn, "clients/client"+str(client_no)+".pkl")

    if client_no == 9 and round_no == 9:
        main = read_object("main.pkl")
        main = aggregation(params["N"], main)
        save_object(main, "main.pkl")