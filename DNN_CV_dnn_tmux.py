#!/home/jason/.conda/environments.txt new
#!/usr/bin/env python3
# coding: utf-8


# In[4]:

import sklearn


# In[5]:


from sklearn.model_selection import KFold
import pickle
import pandas as pd
import os.path
import numpy as np
# path dictionary
path_data_folder = "5.9.23/"
path_dictionary = {
    '20.7': path_data_folder + "twentypointseven",
    '21.0': path_data_folder + "twentyonepointzero_1",
    '21.2': path_data_folder + "twentyonepointtwo_1",
    '21.5': path_data_folder + "twentyonepointfive_1",
    '21.7': path_data_folder + "TwentyonepointseevendegreeC_1",
    '21.8': path_data_folder + "twentypointeight_1"
}
# separator in this file is tab
frames = pd.read_csv("5.9.23/twentypointseven", sep="\t", header=None)
# label for entire file is the temperature





# In[6]:


print(frames.shape[1])
print(frames[9999][999].dtype) # <- pandas index is [column][row]


# In[7]:


frames[0]


# In[21]:


number_of_egs = frames.shape[1]
a = np.zeros((1, number_of_egs)) + 20
df_label = pd.DataFrame(a)
df_label.shape


# In[22]:


df_label.columns = np.asarray(range(df_label.shape[1]))


# In[23]:


df_label[[1, 2, 3]]


# In[8]:


# the data are (1000 * 1) column vectors.
# in the file, there are 1000 lines, each with n numbers, 
# where n = number of data vectors
def load_data(filename_dictionary):
    X_data = []     # data
    y_data = []    # label
    for filename, filepath in filename_dictionary.items():
        print(f"reading file:        {filepath}")
        X_in_this_file = pd.read_csv(filepath, sep="\t", header=None)
        value = float(filename)
        print(f"\ttemperature value: {value}")
        number_of_examples = X_in_this_file.shape[1]
        y_in_this_file = np.zeros(shape=(1, number_of_examples)) + value
        y_in_this_file = pd.DataFrame(y_in_this_file)
        # default column setting is NO array, 
        # need to make it array to use list of indices!
        y_in_this_file.columns = np.asarray(range(y_in_this_file.shape[1]))
        X_data.append(X_in_this_file)
        y_data.append(y_in_this_file)
    X_data = pd.concat(X_data, axis=1, ignore_index=True)
    y_data = pd.concat(y_data, axis=1, ignore_index=True)
    return X_data, y_data


# In[9]:


from sklearn.model_selection import KFold
import pickle
import pandas as pd
# change: response (X) -> spectrum, spectra (y) -> temperature

spectrum, temperature = load_data(filename_dictionary=path_dictionary)
print(f"total number of examples:     {spectrum.shape[1]}")
print(f"length of each example:       {spectrum.shape[0]}")
print(f"shape of X data: {spectrum.shape}, type: {spectrum[0][0].dtype}")
print(f"shape of y data: {temperature.shape}, type: {temperature[0].dtype}")

import os.path
file_name = 'cross_validation_resample=2_fold=5_dnn'
if os.path.isfile(file_name+'.pickle'): 
    with open(file_name+'.pickle', 'rb') as handle:
        train_indices,test_indices = pickle.load(handle)  
else:
    # 2x5 resampling
    train_indices = []
    test_indices = []
    number_resamples = 2
    n_splits =5
    for i in range(number_resamples):
        kf = KFold(n_splits=n_splits, random_state=i, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(range(temperature.shape[1]))):
            train_indices.append(train_index)
            test_indices.append(test_index)
    with open(file_name+'.pickle', 'wb') as handle:
        pickle.dump([train_indices,test_indices], handle, protocol=pickle.HIGHEST_PROTOCOL)
                                            


# In[26]:


temperature.columns


# In[27]:


print(spectrum.shape)
spectrum[[1, 2, 3]]


# In[28]:


print(temperature.shape)
temperature[[1, 2, 3]]


# In[29]:


import torch
import torch.nn as nn
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
print(input.shape)
output = m(input)
print(output.size())


# In[11]:


import torch
import torch.nn as nn
# change: input dim = 1000, output dim = 1 (temperature value)
class Model(torch.nn.Module):
    def __init__(self,device, input_dim=1000):
        super().__init__()
        self.relu  = nn.ReLU()
        self.hidden_dim = 500
        self.linear1 = torch.nn.Linear(input_dim, self.hidden_dim)
        self.linear2= torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3= torch.nn.Linear(self.hidden_dim, 1)
        self.device = device
        self.to(device)
    def forward(self, x):
        y= self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))
        # change: remove sigmoid, use y result as is
        # y = torch.sigmoid(y)
        return y


# In[31]:


from torch.utils.data import DataLoader
from torch import optim
import numpy as np
class CalculateMSE():
    def __init__(self, net, n_epochs, batch_size):
        super().__init__()
        self.net = net
        #initialize some constants
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.n_epochs = n_epochs
        self.net.apply(self.weights_init)   
    def weights_init(self,layer):
        if type(layer) == nn.Linear:
            nn.init.orthogonal_(layer.weight)
    def get_mse(self,train_data, train_label, test_data, test_label):
        train_set = torch.utils.data.TensorDataset(
            torch.Tensor(train_data), 
            torch.Tensor(train_label))
        val_set = torch.utils.data.TensorDataset(
            torch.Tensor(test_data), 
            torch.Tensor(test_label))
        loader_args = dict(batch_size=self.batch_size)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
        tloss = []
        vloss = []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate) # weight_decay=0
        for epoch in range(0, self.n_epochs):
            if epoch % 5 == 0:
                print(f"epoch = {epoch}")
            epoch_train_loss=[]
            for i, data in enumerate(train_loader, 0):
                inputs, label = data
                y_pred = self.net(inputs.to(self.net.device))
                loss = criterion(y_pred, label.to(self.net.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            tloss.append(np.mean(epoch_train_loss))
            epoch_loss=[]
            for i, data in enumerate(val_loader, 0):
                with torch.no_grad():
                    inputs1, label1 = data
                    y_pred1 = self.net(inputs1.to(self.net.device))
                    loss1 = criterion(y_pred1, label1.to(self.net.device))
                    epoch_loss.append(loss1.item())
            vloss.append(np.mean(epoch_loss))
        return np.min(vloss), self.net


# In[ ]:


from pathlib import Path
# change: turn into 10 right now for development, was 3000
n_epochs=3000
batch_size=32
PATH = 'model_dnn/'
Path(PATH).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# change: commented out: alraedy read in X, y data earlier
# response = pd.read_csv("1127_final_data/response.csv", header=None).values #input X
# spectra = pd.read_csv("1127_final_data/spectra.csv", header=None).values #ground truth label Y
# change: input dim = 1000
mdl = Model(device=device, input_dim=1000)
losses = []
for i,(train,test) in enumerate(zip(train_indices,test_indices)):
    print(f"we are on fold no.{i}")
    train_data, train_label= spectrum[train],temperature[train]
    test_data, test_label= spectrum[test],temperature[test]
    mse_calculator = CalculateMSE(mdl,n_epochs,batch_size)
    loss,model = mse_calculator.get_mse(np.transpose(np.asarray(train_data)), 
                                        np.transpose(np.asarray(train_label)), 
                                        np.transpose(np.asarray(test_data)), 
                                        np.transpose(np.asarray(test_label)))
    losses.append(loss)
    print(f"\tloss: {loss}")
    torch.save(model.state_dict(), PATH+'model'+str(i))


# In[ ]:


# takes 78 minutes per fold! loss is 412
# After 700 minute, loss is:
# we are on fold no.0
# 	loss: 412.9017034505208
# we are on fold no.1
# 	loss: 412.9801847330729
# we are on fold no.2
# 	loss: 412.79992659505206
# we are on fold no.3
# 	loss: 412.94208658854166
# we are on fold no.4
# 	loss: 412.96778076171876
# we are on fold no.5
# 	loss: 412.6050344238281
# we are on fold no.6
# 	loss: 412.8362565917969
# we are on fold no.7
# 	loss: 413.03926790364585
# we are on fold no.8
# 	loss: 413.1065719401042
# we are on fold no.9
# 	loss: 413.0045514322917
# intuition, label is about 20, error is MSE, 
# so my predictions are always 0 or 1-ish!
# Problem: used sigmoid(y) at end of neural network!!!


# In[118]:


print(np.mean(losses),np.std(losses))


# In[124]:


len(spectrum)


# In[129]:


print(spectrum[69])


# In[137]:


a = torch.randint(0, 1000,(10,))
a[1].dtype


# In[139]:


spectrum[int(a[1])]


# In[142]:


np.asarray(spectrum[int(a[1])]).flatten()


# In[143]:


np.asarray(temperature[int(a[1])]).flatten()


# In[155]:


number_figures = 10
import matplotlib.pyplot as plt

indices = torch.randint(0,len(spectrum),(number_figures,)).unique()
for i in indices:
    # change: cast i to int, since pandas not work with torch.int64
    spec = np.asarray(spectrum[int(i)]).flatten()
    temp = np.asarray(temperature[int(i)]).flatten()
    plt.figure(i)

    prediction = model(torch.Tensor(np.asarray(spectrum[int(i)])).to(model.device)).detach().cpu().flatten()
    plt.plot(prediction)
    print(prediction.item())
    plt.plot(temp)
    plt.legend(['reconstruction','ground truth'])


# In[3]:


from pathlib import Path
PATH = 'saved_model/DNN/'
device = torch.device("cuda")
mdl = Model(device=device, input_dim=1000)

from scipy import stats,spatial
#pip install dtw-python
from dtw import *
import torch
import numpy as np
# response = pd.read_csv("1127_final_data/response.csv", header=None).values #input X
# spectra = pd.read_csv("1127_final_data/spectra.csv", header=None).values #ground truth label Y
correlation_losses = []

def calculate_correlation(model, test_data, test_label):
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    construction = model(test_data_tensor).detach().cpu().numpy()
   
    # Pearson
    pearson_coefs = []
    pearson_ps = []
    
    # Kendall
    kendall_coefs = []
    kendall_ps = []
    
    # Spearman
    spearman_coefs = []
    spearman_ps = []
    
    # Distance Correlation
    distance_corr = []
    
    #DTW distance
    alignment = []
    
    #absolute_error
    abs_err = []
    
    for i in range(test_label.shape[0]):
        x1 = construction[i,:]
        x2 = test_label[i,:]
        
        res = stats.pearsonr(x1, x2)
        pearson_coefs.append(res[0])
        pearson_ps.append(res[1])
        
        res = stats.kendalltau(x1, x2)
        kendall_coefs.append(res[0])
        kendall_ps.append(res[1])
        
        res = stats.spearmanr(x1, x2)
        spearman_coefs.append(res[0])
        spearman_ps.append(res[1])
        
        distance_corr.append(1- spatial.distance.correlation(x1,x2))
        
        alignment.append(dtw(x1, x2, distance_only=True).distance)
        abs_err.append(abs(x1-x2))
        
    correlation_results = {
        'pearson': (pearson_coefs, pearson_ps),
        'kendall': (kendall_coefs, kendall_ps),
        'spearman': (spearman_coefs, spearman_ps),
        'DTW': alignment,
        'Absolute Error': abs_err,
        'Distance Correlation': distance_corr
    }

    return correlation_results

for i, (train, test) in enumerate(zip(train_indices, test_indices)):
    print(i)
    train_data, train_label = spectrum[train], temperature[train]
    test_data, test_label = spectrum[test], temp[test]
    
    mdl_name = PATH + 'model' + str(i)
    mdl.load_state_dict(torch.load(mdl_name))
    mdl.eval()
    
    correlation_loss = calculate_correlation(mdl, test_data, test_label)
    correlation_losses.append(correlation_loss)
for key in correlation_losses[0].keys():
    print(key)
    if key=='Absolute Error':
        errors = []
        for d in correlation_losses:
            errors+=np.concatenate(d[key]).ravel().tolist()
        #percentile
        percentiles = [5, 50, 90, 95, 99]
        for p in percentiles:
            print(p)
            print(np.percentile(errors, p))
    else:
        stat, p = [], []
        for d in correlation_losses:
            if key=='DTW' or key=='Distance Correlation':
                stat+=d[key]
            else:
                stat+=d[key][0]
                p+=d[key][1]
        print(np.mean(stat),np.std(stat))
        if len(p)>0:
            print(np.mean(p),np.std(p))

