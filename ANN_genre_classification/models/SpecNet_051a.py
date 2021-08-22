import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 11), padding=(2, 5), stride=(2, 9))
        self.act1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.do1 = torch.nn.Dropout2d(0.05)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3)
        
        #1ый дополнительный слой слой
        self.conv1ad = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 9), padding=(2, 4), stride=(3, 5))
        self.act1ad = torch.nn.ELU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=128)
        self.do2 = torch.nn.Dropout2d(0.05)
        self.pool1ad = torch.nn.AvgPool2d(kernel_size=3)

        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.act2 = torch.nn.Tanh()
        self.bn3 = torch.nn.BatchNorm2d(num_features=256)
        self.do3 = torch.nn.Dropout2d(0.05)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1, 2))

        self.fc1 = torch.nn.Linear(256, 64)
        self.fc1_act = torch.nn.ELU()
        
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc2_act = torch.nn.ReLU()
        
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc3_act = torch.nn.Tanh()

        self.fc4 = torch.nn.Linear(16, 8)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.pool1(x)
        
        x = self.conv1ad(x)
        x = self.act1ad(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.pool1ad(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn3(x)
        x = self.do3(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        x = self.fc1(x)  
        x = self.fc1_act(x)

        x = self.fc2(x)
        x = self.fc2_act(x)
        
        x = self.fc3(x)
        x = self.fc3_act(x)

        x = self.fc4(x)


        return x