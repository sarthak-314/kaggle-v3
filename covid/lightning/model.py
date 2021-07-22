from time import time
import torch 

def init_weights(m):
    if type(m) != torch.nn.Linear: return 
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
        
def load_backbone(backbone_name):
    import timm 
    start_time = time()
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    print(f'{round(time()-start_time, 3)} seconds to load backbone')
    return backbone