#from get_data import clients_wind
#from server import Decentralize
import numpy as np
from tqdm import tqdm

import torch
from itertools import chain
from sklearn.metrics import mean_absolute_error, mean_squared_error

#from client import caller
from model import ANN
from client import caller
from dataprocessing import save_object, read_object, nn_seq_wind

def main():
    # N: No.of clients
    # Cper: Percentage of clients to be chosen for every communication round
    # K: No.of update steps in the clients
    # B: Batch size
    # R: No.of communication rounds
    # input_dim: Dimension of the input
    # lr: learning rate

    N, Cper, K, B, R = 10, 0.5, 10, 50, 10
    input_dim = 28
    lr = 0.08

    clients = []
    for task in range(1, 2):
        for zone in range(1, 11):
            clients.append("Task" + str(task) + "_W_Zone" + str(zone))

    params = {"N": N, "Cper": Cper, "K": K, "B": B, "R": R, "clients": clients, "input_dim": input_dim, "lr": lr}

    # save dictionary of default parameters into a .pkl file
    save_object(params, "defaults.pkl")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn = ANN(input_dim = input_dim, name = "main", B = B, K = K, lr = lr).to(device)

    #k= control of fc1` weights`
    for k, v in nn.named_parameters():
        nn.control[k] = torch.zeros_like(v.data)
        nn.delta_control[k] = torch.zeros_like(v.data)
        nn.delta_y[k] = torch.zeros_like(v.data)

    save_object(nn, "main.pkl")

    for r in range(R):
        print("-----------------------------------Round " + str(r+1) + "-----------------------------------")
    
        for i in range(N):
            print("-----------------------------------Client " + str(i) + "-----------------------------------")
            caller(i, r)
    
    #test
    def test(ann):
        ann.eval()
        _, _, Dte = nn_seq_wind(ann.name, ann.B)
        pred = []
        y = []
        for (seq, target) in tqdm(Dte):
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = ann(seq)
                pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
                y.extend(list(chain.from_iterable(target.data.tolist())))

        pred = np.array(pred)
        y = np.array(y)
        print("mae: ", mean_absolute_error(y, pred), "rmse: ", np.sqrt(mean_squared_error(y, pred)))
    
    print("\n\n-------------------Testing the final model on all the clients-------------------\n\n")
    
    model = read_object("main.pkl")
    model.eval()
    
    c = clients
    for client in c:
        model.name = client
        test(model)

if __name__ == '__main__':
    main()