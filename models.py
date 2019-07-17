import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Generator_s2sa(nn.Module):
    def __init__(self):
        super(Generator_s2sa, self).__init__()
        self.conv1 = Conv_Block(1, 64, kernel_size=5)    
        self.conv2 = Conv_Block(64, 64, kernel_size=5) 
        self.conv3 = Conv_Block(64, 128, kernel_size=5, stride=2)
        self.fc1 = Dense_Block(5888, 1000)
        self.fc2 = Dense_Block(1000, 100)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Classifier_s2sa(nn.Module):
    def __init__(self, n_output):
        super(Classifier_s2sa, self).__init__()
        self.fc = nn.Linear(100, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self, task='s2sa'):
        super(Net, self).__init__()
        if task == 's2sa':
            self.generator = Generator_s2sa()
            self.classifier = Classifier_s2sa(6)
        else:
            print('Error in nn.modules')
			
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, adaption = False):
        x = self.generator(x)
        if adaption == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x