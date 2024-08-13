import torch
import torch.nn as nn

class Conv1D_model(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv1D_model, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(22336, 512)  
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor to 1D
        x = self.fc1(x)
        
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        
        x = self.fc2(x)
        
        return x
    
    def forward_features(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1) 

        return x
    
#############################################################################################
#############################################################################################

class Conv_LSTM(nn.Module):
    def __init__(self, num_classes=2):
        #In the paper num_class = 4
        super(Conv_LSTM, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM
        self.lstm = nn.LSTM(input_size= 11168,
                            hidden_size= 64,
                            batch_first= True,
                            dropout= 0,
                            bidirectional= True)


        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 256)
        #self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(p = 0.5)

        self.fc2 = nn.Linear(256, 128)
        #self.relu4 = nn.ReLU()
        self.drop2 = nn.Dropout(p = 0.5)

        self.fc3 = nn.Linear(128, num_classes)

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  
        
        x = x.unsqueeze(1)  
        x,_ = self.lstm(x)
        x = x.view(x.size(0), -1)  

        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        x = self.fc3(x)


        return x
    
    def predict(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1) 
        
        x = x.unsqueeze(1)  
        x,_ = self.lstm(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        x = self.fc3(x)

        return x
    
    def forward_features(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  

        x = x.unsqueeze(1)  
        x,_ = self.lstm(x)
        x = x.squeeze()
        
        return x
    
################################################################################################
################################################################################################

import torch.nn as nn

class Fus_model(nn.Module):
    def __init__(self, num_classes=2):
        super(Fus_model, self).__init__()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(2 * num_classes, num_classes)  
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)

        return x