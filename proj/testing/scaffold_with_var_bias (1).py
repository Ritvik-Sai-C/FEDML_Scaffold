#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary libraries for the code
import numpy as np  #used mainly to save and load data from .npz files (our data in the drive is also stored in this format)
import matplotlib.pyplot as plt #used for plotting or data visualization. In this code the part is all commented (the last 3-4 cells)
import copy #used to make deepcopy of a neural network (meaning of deepcopy is explained in the cell below)
import IPython  #IPython is an interactive command-line terminal for Python (but it is not used in any of the running code)
from PIL import Image #used to perform tasks with images (but it is not used in running code)

#every library below is used to tp create, train and evaluate the neural networks we create for our data
#the use of each library is explained when it is used in the code
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# In[ ]:


'''
Difference between copy and deepcopy:
Copy/Shallow copy - creating a new pointer but area of data is the same
y = copy.copy(x) which visually means y -> □ <- x (both x and y are pointing to the same data block)
Hence, x = 5 changes the value of y as well to 5

Deepcopy - creating a new block of data with new pointer but same information
y = copy.deepcopy(x) which visually means x -> □ and y -> □ (separate copy of data is created)
Hence, x = 5 doesn't affect the data in y
'''


# In[ ]:


# set random seeds (this is used to ensure that you get the same randomness everytime no matter how many times you run the code)
np.random.seed(0)
torch.manual_seed(0)

# set device (if gpu/cuda is available perform the neural network operations using that else use a cpu)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("| using device:", device)


# In[ ]:


# hyperparameters
bsz = 10  #batch size
no_clients = 20 #no.of clients
lamda = 0.7 #not used anywhere in the code
epsilon = 1e-10 #used in scaffold_experiment function (not sure what formula is used)


# In[ ]:


#mounting drive to fetch noniid data from drive location
#to make this work, first upload the data folder into your drive (only then data can be accessed)
from google.colab import drive
drive.mount('/content/drive',force_remount=True)


# In[ ]:


#a class NonIIDMyDataset is created to access and transform data
class NonIIDMyDataset(Dataset):
    #the __init__ function in Python is like the C++ constructor in an object-oriented approach
    def __init__(self, data, targets, transform=None):
        self.data = data  #data is X
        self.targets = torch.LongTensor(targets)  #tragets are y. Convert y to a tensor to be able to used torch function on them
        self.transform = transform  #this is the transformation to be applied on to X. By default the value is None.
                                    #In the 2nd cell below you can see the exact transform used in the code

    #this function is used to apply a transformation (if any) to X and return the pair (X, y) based on the index passed
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            # x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)

        return x, y

    #this function is used to get the length/no.of features of X
    def __len__(self):
        return len(self.data)

#The functions with __ at the front and back of every function in Python are called Magic Methods/Dunder Methods.
#Magic methods are not meant to be invoked directly by you, but the invocation happens internally from the class on a certain action.
#Not sure of the meaning but may be this concept is understood better if you find where these methods are used int he code.


# In[ ]:


#these are the locations of train and test data for 20 clients used in the code
#there are many other folders as well in this dataset folder. May be different ones are used for different cases
train_location = '/content/drive/MyDrive/dataset/practical/20/train/'
test_location = '/content/drive/MyDrive/dataset/practical/20/test/'


# In[ ]:


#the transforms library imported above, is used to create a transformation of the data(X)
#transforms.Compose - to put more than one sequantial transforms into one
#transforms.ToTensor - to convert a list/np array to a tensor
#transform.Normalize - transforms.Normalize(mean, std, inplace=False) to normalize a tensor with give mean and std
#to normalize a data means changinf x to (x-mean)/std
#here mean is 0.137 and std is 0.3081. May be these values are obtained by calculating mean and std of X separately or they are random. Not sure

#how did these value we got
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

client_loader = []  #this list is used to store the train data loaded using 'DataLoader' module from torch in batches

#this function converts y of data to a tensor using __init__, applies the above transformation to x using the __getitenm__ function in the NonIIDMyDataset
#and loads the data in batches (in order to train it with a neural network later) and stores the loaded data into client_loader list created above
def noniid_train_loader(bsz=10):
  #for all the 20 clients
  for i in range(no_clients):
    #go to the folder /content/drive/MyDrive/dataset/practical/20/train/, read the file from client_num.npz (liek 1.npz, 2.npz ... 20.npz) and store the X and y values
    file_path = str(i)+'.npz'
    loc = train_location + file_path
    data = np.load(loc)
    X = list(data['features'])
    Y = list(data['labels'])

    #create an object called dataset which is an instance of the class NonIIDMyDataset
    dataset = NonIIDMyDataset(X, Y, transform=transform)
    #in batches of 10, load the whole dataset and store it in client_load
    client_load = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)

    #append every client's dataload into client_loader list
    client_loader.append(client_load)

  print(client_loader)  #you can see 20 objects of torch dataloaders
  return client_loader


# In[ ]:


noniid_client_train_loader = noniid_train_loader(bsz = bsz) #call the above funtion to perform all the actions explained inside the func, noniid_train_loader


# In[ ]:


#the exact same thing as in the func noniid_train_loader is done here. Expect that the data is extracted now read from the loacation /content/drive/MyDrive/dataset/practical/20/test
test_loader = []
def noniid_test_loader(batch_size,shuffle):
  for i in range(no_clients):
    file_path = str(i)+'.npz'
    loc = test_location + file_path
    data = np.load(loc)
    X = list(data['features'])
    Y = list(data['labels'])

    dataset = NonIIDMyDataset(X, Y, transform=transform)
    client_load = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)

    test_loader.append(client_load)

  print(test_loader)
  return test_loader


# In[ ]:


test_loader = noniid_test_loader(batch_size = 1000, shuffle=False)  #test data is tranformed loaded in batches of 1000 and stored in test_loader


# In[ ]:


# non-iid
#this cell is totally just for observation
label_dist = torch.zeros(10)  #since we have 10 classes, create a torch array with 10 zeros
print(type(noniid_client_train_loader[0]))
print(noniid_client_train_loader[0])

#using a for-loop, count the no.of rows is dataset which has 10 classes respectively for client 1
for (x,y) in noniid_client_train_loader[0]:
    label_dist+= torch.sum(F.one_hot(y,num_classes=10), dim=0)  #one-hot encoding is explained int he next cell

print("non-iid: ", label_dist)
#I suppose there should be a line like label_dist = torch.zeros(10) here as well
for (x,y) in test_loader[0]:

    label_dist+= torch.sum(F.one_hot(y,num_classes=10), dim=0)

print("non-iid: ", label_dist)
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# for i, ax in enumerate(axes.flat):
#     ax.axis("off")
#     ax.set_title(label[i].cpu().numpy())
#     ax.imshow(img[i][0], cmap="gray")
# IPython.display.display(fig)
# plt.close(fig)


# In[ ]:


'''
one hot encoding is a concept where we assign 1 for the class of that row and 0 for the rest
example say we have 5 classes in the dataset.
The classes of say 10 rows of data are 1 3 2 4 1 5 3 2 1 4. (i.e., 1st row of data belongs to class 1 ...)
After applying one hot encoding the classes of these 10 rows will be represented as
1th row : 1 0 0 0 0
2th row : 0 0 1 0 0
3th row : 0 1 0 0 0
4th row : 0 0 0 1 0
5th row : 1 0 0 0 0
6th row : 0 0 0 0 1
7th row : 0 0 1 0 0
8th row : 0 1 0 0 0
9th row : 1 0 0 0 0
10th row: 0 0 0 1 0
'''


# In[ ]:


#this function is only used to observe how many parameters are used in the neural network we create. It is only for observation. Not to effect the running of any code
#parameters in neural networks are like no.of weights or bias params included to the network. Check it out on the internet
def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[ ]:


# define fully connected NN
#this version of Multi Layer Perceptron is not to train data in the code. Instead, the CNN defined in the next cell is used
#the network here is: x -> linear layer -> relu activation -> linear layer -> relu activation -> linear -> output
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200);
        self.fc2 = nn.Linear(200, 200);
        self.out = nn.Linear(200, 10);

    def forward(self, x):
        x = x.flatten(1) # [B x 784]
        x = F.relu(self.fc1(x)) # [B x 200]
        x = F.relu(self.fc2(x)) # [B x 200]
        x = self.out(x) # [B x 10]
        return x

print(MLP())
print(num_params(MLP()))


# In[ ]:


# define cnn
#A CNN (Convolutional Neural Network) is another kind of NN.
#In MLPs, there are layers like linear layers where a linear operation like y = w.T*x+b is applied (a linear operation) followed by activation
#Similarly, in CNN, as the name suggests, convolution is done on x (input) to get y (output) on some layers. Here kernels are used.
#I suggest you to look through some blogs and understand practically and mathematically

#the network below is: input -> convolution 2D layer -> max pool activation -> conv 2d -> max pool -> linear -> relu -> linear -> output
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2) # [B x 32 x 12 x 12]
        x = F.max_pool2d(self.conv2(x), 2, 2) # [B x 64 x 4 x 4]
        x = x.flatten(1) # [B x 1024]
        x = F.relu(self.fc(x)) # [B x 512]
        x = self.out(x) # [B x 10]
        return x

print(CNN())
print(num_params(CNN()))


# In[ ]:


'''
The whole idea of neural network and data revolves around the below steps:
1. Create a basic neural network be it MLP, CNN, RNN
2. Transform & Normalize data to be able to train and validate data using the network
3. Change the weights etc., parameters of the neural network through back propogation or any other method
4. For the same choose a loss function and an optimizer.
5. Repeat until you reach some fixed no.of iterations or desired result

So basically train your network with initial weights and the data and predict ŷ.
Calculate loss/error using the loss func you choose. An example is (y-ŷ).
If the error is more, re-train the network with new weights. This is done through back propgation which is automatically done most of the times.
'''


# In[ ]:


criterion = nn.CrossEntropyLoss() #the loss function we chose is cross entropy. The mathematical formula is available on the internet

#the below function is used to validate (find the percentage of correctly predicted output)
def validate(model,client):
    #change the model/network to evaluation mode and for the given client, predict ŷ = model(x). If ŷ=y, add 1 to correct
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (t, (x,y)) in enumerate(test_loader[client]):
            x = x.to(device)
            x = x.permute(0, 2, 3, 1)
            #print("x",x.shape)
            y = y.to(device)
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.shape[0]
    return correct/total


# In[ ]:


def train_client(id, client_loader, global_model, num_local_epochs, lr):
    #create a deepcopy of the global model and change the network to train mode
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    #for given no.of epochs (iterations) run the for-loop on the client
    for epoch in range(num_local_epochs):
        #for every pair of (X, y), predict ŷ for X using the local model, find the loss (using cross entropy loss here) which is l
        #predict ŷ for X using the global model, find the loss with this ŷ and y which is dl
        #I guess this is some new formula in this version. find the total loss for present (X, y) using the formula
        #(1-p)*loss + p*drift_loss where p = l/(l+dl+epsilon)
        #with this loss, as said in the 2nd cell above, we perform back propogation and optimize.
        #I am exactly not sure of the math or steps which happen in back prop and optimization
        #You can check it online. You may get an idea
        for (i, (x,y)) in enumerate(client_loader):
            x = x.to(device)

            #x.reshape(10, 1, 28, 28)
            x = x.permute(0, 2, 3, 1)
            #print("x",x.shape)
            y = y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            global_output_batch = global_model(x)
            drift_loss = criterion(global_output_batch,y)
            l = loss.item()#print('loss',loss.item())
            dl = drift_loss.item()
            #print('drift loss',drift_loss.item())
            p = l/(l+dl+ epsilon)
            #print(p)
            total_loss = (1-p)* loss+ p*drift_loss
            total_loss.backward()
            optimizer.step()

    #after training the local model with the client's data for given no.of epochs, we return the local model
    #So here, the overall model (like the server model) as effects the weights/params of local model
    #cx every client has their own global model and local model like in our prev sem code
    #remember we used to approach to remove the usage of server completely?
    return local_model

#this func is same as aggregation func in the prev sems code. models argument is an array of local models of all the clients
def running_model_avg(models,client_no):
    # Initialize the aggregated model's state dictionary
    aggregated_state_dict = models[client_no].state_dict()
    it = 0
    # Iterate over the models of the clients
    for client_model in models:
        #if the clinet_model is not the one passed as an argument
        if it!=client_no:
          # Get the state dictionary of the current client model
          client_state_dict = client_model.state_dict()

          # Iterate over the parameters in the state dictionary
          for param_name, param in client_state_dict.items():
              # Perform aggregation for each parameter
              if param_name.endswith(".weight") or param_name.endswith(".bias"):
                  # Update the aggregated parameter by averaging
                  aggregated_state_dict[param_name] += param
        it = it + 1

    # Compute the average by dividing by the number of client models
    num_client_models = len(models)
    for param_name in aggregated_state_dict:
        aggregated_state_dict[param_name] /= num_client_models

    # Create a new model instance for the aggregated model
    aggregated_model = type(models[0])()
    aggregated_model.load_state_dict(aggregated_state_dict)

    return aggregated_model


# In[ ]:


'''
Basically what is happening above is that, create a state_dict called aggregated_state_dict and intitialize it with our present client's data
Now run a for loop through all the other client's model and add their params (state_dict) to the aggregated_state_dict
Then to normalize it, divide all the params of aggreated_state_dict by no.of clients (20 here)
Then create a model structure for aggregated_model and load all these params into this.

But ig we can just add the params of all clients directly instead of that if it!=client_no statement. Cz at the end I feel we are just adding the params of all the local models.
'''


# In[ ]:


#modified scaffold each client having their own global model global_model[0] for client[0]

def scaffold_experiment(global_model, num_clients, num_local_epochs, lr, client_train_loader, max_rounds, filename):   ##num_client
    round_accuracy = []
    #for all the rounds
    for t in range(max_rounds):
        print("starting round {}".format(t))

        # choose clients
        #clients = np.random.choice(np.arange(100), num_clients, replace = False)  ###remove this
        clients=num_clients
        print("clients: ", clients)   ##
        #create 2 list to store global (aggregated) and local models of each client
        running_avg = [None for _ in range(clients)]
        local_models = [None for _ in range(clients)]

        for i in range(clients):
          global_model[i].eval()
          global_model[i] = global_model[i].to(device)
          #running_avg = np.empty(clients, dtype='collections.OrderedDict') #None

        #for all the clients, train their local models with their dataset
        for i in range(clients):
            # train local client
            print("round {}, starting client {}/{}, id: {}".format(t, i+1,num_clients, i+1))
            local_models[i] = train_client(i, client_train_loader[i], global_model[i], num_local_epochs, lr)

        # add local model parameters to running average of each client
        for j in range(clients):
          #print(type(running_avg[j]), '   and ',type(local_model.state_dict()))
          global_model[j] = running_model_avg(local_models,j)


       # validate
        acc_val = 0
        for client in range(no_clients):
          val_acc = validate(global_model[client],client)
          #print('each client',val_acc)
          acc_val = acc_val + val_acc
        acc_val = acc_val/no_clients
        print("round {}, validation acc: {}".format(t, val_acc))
        round_accuracy.append(acc_val)

        if (t % 10 == 0):
          np.save(filename+'_{}'.format(t)+'.npy', np.array(round_accuracy))

    return np.array(round_accuracy)


# In[ ]:


#cnn modified cnn_iid_m10 for each client seperate
cnn = CNN() #create a CNN object
print(cnn)
print("total params: ", num_params(cnn))
# CNN - iid - m=10 experiment
cnn_noniid_r10_ep10 = np.empty(no_clients, dtype=CNN) #create a CNN with uninitialized and random params (weights and bias) with size 20

#for all the 20 clients, make a deepcopy of cnn created in the first line of this cell (initializing the empty array created above)
#these will be the global models for all the 20 clients
for i in range(no_clients):
  cnn_noniid_r10_ep10[i] = copy.deepcopy(cnn)

#cnn_iid_m10 = copy.deepcopy(cnn)
#calculate the accuracies obtained by the final aggregated global model for all the client's datasets
#we are passing, global models of all clients, no.of clients, no.of epochs, learning rate for optimizer,
#the train data of all clients loaded (at the beginning of this notebook), no.of rounds to run the exp, filepath to store the results
acc_cnn_noniid_r10_ep10 = scaffold_experiment(cnn_noniid_r10_ep10, num_clients=no_clients,
                                 num_local_epochs=5,
                                 lr=0.01,
                                 client_train_loader = noniid_client_train_loader,
                                 max_rounds=30,  ## for testing keep it 1
                                 filename='/content/drive/MyDrive/dataset/results/acc_cnn_noniid_clt20_e20_biasvar')
print(acc_cnn_noniid_r10_ep10)
np.save('/content/drive/MyDrive/dataset/results/acc_cnn_noniid_clt20_ep20_biasvar.npy', acc_cnn_noniid_r10_ep10)


# In[ ]:


#this function is not use anywhere in the code
def view_10(img, label):
    """ view 10 labelled examples from tensor"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(label[i].cpu().numpy())
        ax.imshow(img[i][0], cmap="gray")
    IPython.display.display(fig)
    plt.close(fig)


# In[ ]:


# acc_cnn_noniid_r10_ep10 = np.load('/content/drive/MyDrive/dataset/results/acc_cnn_noniid_r10_ep10_bias9.npy')


# In[ ]:


# x = np.arange(0,15)
# plt.figure(figsize=(8,6))

# plt.title("Scaffold test accuracy after $t$ rounds on non-iid MNIST")

# plt.xlabel("Communication rounds $t$")
# plt.ylabel("Test accuracy")
# plt.axis([0, 15, 0.3, 1])



# plt.axhline(y=0.7, color='r', linestyle='dashed')
# plt.axhline(y=0.9, color='b', linestyle='dashed')

# plt.plot(x, acc_cnn_noniid_r10_ep10, label='2NN, $m=10$, $E=1$')


# In[ ]:


# acc_cnn_noniid_r10_ep10_b = np.load('/content/drive/MyDrive/dataset/results/acc_cnn_noniid_r10_ep10_bias8.npy')
# acc_cnn_noniid_r10_ep10_wb = np.load('/content/drive/MyDrive/dataset/results/acc_cnn_noniid_r10_ep10_nobias.npy')
# acc_cnn_noniid_r10_ep10_b9 = np.load('/content/drive/MyDrive/dataset/results/acc_cnn_noniid_r10_ep10_biasvar.npy')


# In[ ]:


# x = np.arange(0,15)
# plt.figure(figsize=(8,6))

# plt.title("Scaffold test accuracy after $t$ rounds on non-iid MNIST")

# plt.xlabel("Communication rounds $t$")
# plt.ylabel("Test accuracy")
# plt.axis([0, 15, 0.3, 1])



# plt.axhline(y=0.5, color='r', linestyle='dashed')
# plt.axhline(y=0.9, color='b', linestyle='dashed')

# graph1, =plt.plot(x, acc_cnn_noniid_r10_ep10_b, label='b8')
# graph2, =plt.plot(x, acc_cnn_noniid_r10_ep10_wb, label='wb')
# graph3, =plt.plot(x, acc_cnn_noniid_r10_ep10_b9, label='b9')
# plt.legend(handles=[graph1, graph2, graph3],loc ="lower right")
# #plt.legend(["blue", "green","red"], loc ="lower right")


# plt.plot(x, acc_cnn_noniid_r10_ep10_wb, label='2NN, $m=10$, $E=1$')

# 
