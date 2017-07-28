# load and convert to GPU model to CPU:

```
import torch
from torchvision import models
import torch.nn.parallel
model = models.AlexNet()
checkpoint = torch.load('model_best.pth.tar')
state_dict = checkpoint['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[0:9] + k[16:] # remove `module.`
    if k[0] == 'f':
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
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
