import warnings
import torch
import random
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

class NeuralWatermark:
    def __init__(self, model=None, mark_type="C", device="cpu", trigger_loader=None, image_size=None, trigger_size=50, num_labels=None, batch_size=1) -> None:
        self.model = model
        self.mark_type = mark_type
        self.device = device
        self.trigger_loader = trigger_loader
        self.image_size = image_size
        self.trigger_size = trigger_size
        self.num_labels = num_labels
        self.batch_size = batch_size

    def get_watermark_keys(self):
        if self.mark_type == "C":
            assert self.trigger_loader != None, "For custom watermarking type, trigger dataset loader cannot be empty"
            return self.get_trigger_keys()

        elif self.mark_type == "N":
            assert self.image_size != None, "For noise watermarking, image size has to be provided"
            assert self.num_labels != None, "For noise watermarking, number of labels has to be provided"
            return self.get_noise_keys()


    def get_trigger_keys(self):
        X, Y = [], []
        for _, data in enumerate(self.trigger_loader):
            inputs, labels = data
            X += list(inputs.cpu().detach().numpy())
            Y += list(labels.cpu().detach().numpy())
            
        custom_keys = {}
        custom_keys['inputs'] = X
        custom_keys['labels'] = Y
        custom_keys['bounds'] = (min(Y), max(Y))


        return custom_keys

    
    def get_noise_keys(self):
        
        X = torch.randn([self.trigger_size] + list(self.image_size))
        labels = [int(i) for i in range(self.num_labels)]
        Y = random.choices(labels, k=self.trigger_size)

        ownership = {}
        ownership['inputs'] = X
        ownership['labels'] = Y
        ownership['bounds'] = (min(Y), max(Y))

        trigger_set = []

        for x, y in zip(X, Y):
            trigger_set.append((x,y))

        self.trigger_loader = DataLoader(trigger_set, batch_size=self.batch_size, shuffle=True)

        return ownership
        


    def get_watermark_loss(self, criterion=None):
        assert criterion != None, "For custom watermarking type, trigger dataset loader cannot be empty"
        wmloss = 0
        wmloss = self.calculate_watermark_loss(criterion)
        return wmloss

    def calculate_watermark_loss(self,criterion ):  
        markedloss = None
        for _, data in enumerate(self.trigger_loader):
            inputs, labels = data
            outputs = self.model(inputs.to(self.device))
            if not markedloss:
                markedloss = criterion(outputs, labels.to(self.device))
            else:
                markedloss += criterion(outputs, labels.to(self.device))
        return markedloss    