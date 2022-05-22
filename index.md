## Doumentation

### 1. Install package
```console
pip install neuralnetwork
```
### 2. Import
```python
from neuralwatermark.neuralwatermark import NeuralWatermark
```
### 3. NeuralWatermark()
These are the parameters for watermark class(NeuralWatermark)
- model : model instance that is being trained, default = None 
- mark_type : watermarking type, "C" for watermarking using images(triggers), default = "C" 
- device : device to use for embedding watermarks, default = "cpu"
- trigger_loader : dataloader with triggers(trigger images), if mark_type is "C"

### 4. Available functions
- get_watermark_keys() :  call to get watermarked data (used to verify model, in future) 
- get_watermark_loss() :  call to get learning loss of the model during watermarking

### 5. Usage
```python
# Initialise the watermark instance 

self.watermark_object = NeuralWatermark(model= self.model, mark_type="C", trigger_loader=self.triggerloader)
# self.model = your model
# self.trigger_loader = trigger dataloader
 

# embed the watermark 
 
 def train_step()
     ...
     wmloss = self.watermark_object.get_watermark_loss(self.args.criterion)
     loss += wmloss
     ...
 # self.args.criterion = your loss criterion 
 
 
 # Save the watermark 
 
 def train()
     ...
     ownership = self.watermark_object.get_watermark_keys()
     with open('ownership.pickle', 'wb') as handle:
        pickle.dump(ownership, handle, protocol=pickle.HIGHEST_PROTOCOL)  
     return ownership 
 ```
 ---------------------------------------
### Work in progress
- Currently the package only supports pytorch, extend it to popular libraries
- Currently the package only supports image classification, expand it to different domains 
- Currently the package only supports watermarking using trigger images, expand using different watermarking techniques
