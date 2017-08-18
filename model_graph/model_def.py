import torch.nn as nn
import torch.nn.functional as f


class ModelDef(nn.Module):

    def __init__(self, num_classes=1000):
        super(ModelDef, self).__init__()
        self.conv1=nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.bn=nn.BatchNorm2d(3)
        self.fc=nn.Linear(3*224*224,1000)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1(x)
        residual = x
        x = self.bn(x)
        x=self.relu(x)
        x+=residual
        x = x.view(-1, 3*224*224)
        x = self.fc(x)

        return x
