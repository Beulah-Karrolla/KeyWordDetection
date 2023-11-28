import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class TempSpeechClassifier(nn.Module):
    def __init__(self):
        super.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32*126*126, 128)
        self.fc2 = nn.Linear(128,1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
class SpeechClassifierModel(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(SpeechClassifierModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.device = device    
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout, 
                            bidirectional=bidirectional, batch_first=True)
        
        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        out,(hn,cn) = self.lstm(x)
        out = self.fc(out)
        #out = nn.AvgPool1d(1)(out)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0,2,1))
        out = torch.sigmoid(out)
        return out
    
class SpeechClassifierModelTransformer(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(SpeechClassifierModelTransformer, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.device = device
        #self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_size*self.direction, nhead=4, dim_feedforward=256, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dim_feedforward=256, dropout=0.2)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        out = self.transformer(x)
        out = self.fc(out)
        #out = nn.AvgPool1d(1)(out)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0,2,1))
        out = torch.sigmoid(out)
        return out
    
        ''' _, (hn, _) = self.lstm(x)
        hn = hn.transpose(0,1)
        _, (hn2, _) = self.lstm2(hn)
        hn2 = hn2.transpose(0,1)
        _, (hn3, _) = self.lstm3(hn2)
        out = self.fc(hn3)
        out = torch.sigmoid(out[-1])
        return out'''

    # Example usage:
    '''num_classes = 1
    feature_size = 40
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    bidirectional = True

    model = SpeechClassifierModel(num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional)

'''
