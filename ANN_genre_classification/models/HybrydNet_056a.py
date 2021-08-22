import torch


class HybrydNet(torch.nn.Module):
    def __init__(self):
        super(HybrydNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 11), padding=(3, 5), stride=(3, 5))
        self.act1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.do1 = torch.nn.Dropout2d(0.05)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 3))
        
        self.conv1ad = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 7), padding=(2, 5), stride=(3, 3))
        self.act1ad = torch.nn.ELU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=128)
        self.do2 = torch.nn.Dropout2d(0.05)
        self.pool1ad = torch.nn.AvgPool2d(kernel_size=(1, 3))
        
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.act2 = torch.nn.Tanh()
        self.bn3 = torch.nn.BatchNorm2d(num_features=256)
        self.do3 = torch.nn.Dropout2d(0.05)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv2ad = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 3), padding=(2, 1))
        self.act2ad = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm2d(num_features=512)
        self.do4 = torch.nn.Dropout2d(0.05)
        self.pool2ad = torch.nn.AvgPool2d(kernel_size=5, stride=3, padding=2)
        
        self.fc1 = torch.nn.Linear(3072, 1024)
        self.fc1_act = torch.nn.ELU()
        
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc2_act = torch.nn.ReLU()
        
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc3_act = torch.nn.Tanh()

        self.fc4 = torch.nn.Linear(64, 8)
    
    def forward(self, x):
        x = x.unsqueeze(0)
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
        
        x = self.conv2ad(x)
        x = self.act2ad(x)
        x = self.bn4(x)
        x = self.do4(x)
        x = self.pool2ad(x)
        
        
        
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        # print(x.shape)
        try:
            x = self.fc1(x)
            x = self.fc1_act(x)

            x = self.fc2(x)
            x = self.fc2_act(x)
            
            x = self.fc3(x)
            x = self.fc3_act(x)

            x = self.fc4(x)
        except Exception as exc:
            print(x.shape)
            raise exc

        return x