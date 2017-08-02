# load and convert to GPU model to CPU:

```
# load model and convert model to cpu:                                                                                      
import torch                                                                                                                
from torchvision import models                                                                                              
import torch.nn.parallel                                                                                                    
model = models.AlexNet()                                                                                                    
checkpoint = torch.load('model_best.pth.tar')                                                                               
state_dict = checkpoint['state_dict']                                                                                       
print('loaded state dict:', state_dict.keys())                                                                              
                                                                                                                            
print('\nIn state dict keys there is an extra word inserted by model parallel: "module.". We remove it here:')              
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
                                                                                                                            
print('Now see converted state dict:')                                                                                                        
print(new_state_dict.keys())                                                                                                
                                                                                                                            
# saving model:                                                                                                             
model_dict={}                                                                                                               
model_dict['model_def']=model                                                                                               
model_dict['weights']=model.state_dict()                                                                                    
torch.save(model_dict, 'model_cpu.pth') 
```

See: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3

And: https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032


# example load 
```
model_dict = torch.load('model_cpu.pth')
model = model_dict['model_def']
model.load_state_dict( model_dict['weights'] )
```
