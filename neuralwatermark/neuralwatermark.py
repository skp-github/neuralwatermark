import warnings
import torch

warnings.filterwarnings('ignore')

class NeuralWatermark:
    def __init__(self, model=None, mark_type="C") -> None:
        self.model = model
        self.mark_type = mark_type
    
    def get_watermark_keys(self, trigger_loader=None):
        if self.mark_type == "C":
            assert trigger_loader == None, "For custom watermarking type, trigger dataset loader cannot be empty"
            return self.get_custom_keys(trigger_loader)
        
    def get_custom_keys(self, trigger_loader):
        X, Y = [], []
        for _, data in enumerate(trigger_loader):
            inputs, labels = data
            temp_x = inputs.cpu().detach().numpy()
            temp_y = labels.cpu().detach().numpy()
            X.append(temp_x)
            Y.append(temp_y)

        custom_keys = {}
        custom_keys['inputs'] = X
        custom_keys['labels'] = Y
        custom_keys['bounds'] = (min(X), max(Y))

        return custom_keys

        