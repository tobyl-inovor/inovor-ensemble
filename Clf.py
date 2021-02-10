import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Clf(nn.Module):
    
    def __init__(self):
        super(Clf, self).__init__()

        # # average pooling before classification layer is optional - required for efficientnet but not for mobilenet-v2
        # if avg_pool:
        #     self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        # else:
        #     self.avg_pooling = Identity()
        
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_features=1280, out_features=2, bias=True)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x = self.avg_pooling(x)
        x = self.dropout(x.squeeze())
        x = self.fc(x)
        
        return x