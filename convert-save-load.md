

# load and convert to GPU model to CPU:

```
import torch
from torchvision import models
import torch.nn.parallel
model = models.AlexNet()
model.features = torch.nn.DataParallel(model.features)
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.cpu()
```

# convert to list of model and weights and save:
```
model_dict={}
model_dict['model_def']=model
model_dict['weights']=model.state_dict()
torch.save(model_dict, 'model_cpu.pth')
```

# example load 
```
model_dict = torch.load('model_cpu.pth')
model = model_dict['model_def']
model.load_state_dict( model_dict['weights'] )
```
