# ESG-BERT
Domain Specific BERT Model for Text Mining in Sustainable Investing

The further pre-trained ESG-BERT model can be found here at this GitHub repository. It is a PyTorch model but it can be converted into a Tensorflow model. They can be fine-tuned using either framework. I found the PyTorch framework to be a lot cleaner, and easier to replicate with other models. However, serving the final fine-tuned model is a lot easier on TensorFlow, than on PyTorch. 

For fine-tuning the model, you can use this command to load it into PyTorch. 
```
model = BertForSequenceClassification.from_pretrained(
    'path/to/dir/containing/ESG-BERT', 
    num_labels = num, #number of classifications
   output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.to(device)

```
The fine-tuned model for text classification is also available on this repository. It can be used directly to make predictions using just a few steps. 
First, download the fine-tuned pytorch_model.bin, config.json, and vocab.txt into your local directory. Make sure to place all of them into the same directory, mine is called "bert_model". 
JDK 11 is needed to serve the model. Go ahead and install it from the Oracle downloads page. Now we are ready to set up TorcheServe.
TorchServe is a model serving architecture for PyTorch models, go ahead and install that using pip. You can also use conda for the installation. 
```
pip install torchserve torch-model-archiver

```
Next up, we'll set up the handler script. It is a basic handler for text classification that can be improved upon. Save this script as "handler.py" in your directory. [6]
```
from abc import ABC
import json
import logging
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
logger = logging.getLogger(__name__)
class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False
def initialize(self, ctx):
        self.manifest = ctx.manifest
properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
# Read model serialize/pt file
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
self.model.to(self.device)
        self.model.eval()
logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
# Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')
self.initialized = True
def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)
inputs = self.tokenizer.encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs
def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.
        prediction = self.model(
            inputs['input_ids'].to(self.device), 
            token_type_ids=inputs['token_type_ids'].to(self.device)
        )[0].argmax().item()
        logger.info("Model predicted: '%s'", prediction)
if self.mapping:
            prediction = self.mapping[str(prediction)]
return [prediction]
def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output
_service = TransformersClassifierHandler()
def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
if data is None:
            return None
data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)
return data
    except Exception as e:
        raise e

```
TorcheServe uses a format called MAR (Model Archive). We can convert our PyTorch model to a .mar file using this command:
```
torch-model-archiver --model-name "bert" --version 1.0 --serialized-file ./bert_model/pytorch_model.bin --extra-files "./bert_model/config.json,./bert_model/vocab.txt" --handler "./handler.py" 
```
Move the .mar file into a new directory: 
```
mkdir model_store && mv bert.mar model_store 
```
Finally, we can start TorchServe using the command: 
```
torchserve --start --model-store model_store --models bert=bert.mar
```
We can now query the model from another terminal window using the Inference API. We pass a text file containing text that the model will try to classify. 

```
curl -X POST http://127.0.0.1:8080/predictions/bert -T predict.txt
```
This returns a label number which correlates to a textual label. This is stored in the label_dict.txt dictionary file. 
```
__label__Business_Ethics :  0
__label__Data_Security :  1
__label__Access_And_Affordability :  2
__label__Business_Model_Resilience :  3
__label__Competitive_Behavior :  4
__label__Critical_Incident_Risk_Management :  5
__label__Customer_Welfare :  6
__label__Director_Removal :  7
__label__Employee_Engagement_Inclusion_And_Diversity :  8
__label__Employee_Health_And_Safety :  9
__label__Human_Rights_And_Community_Relations :  10
__label__Labor_Practices :  11
__label__Management_Of_Legal_And_Regulatory_Framework :  12
__label__Physical_Impacts_Of_Climate_Change :  13
__label__Product_Quality_And_Safety :  14
__label__Product_Design_And_Lifecycle_Management :  15
__label__Selling_Practices_And_Product_Labeling :  16
__label__Supply_Chain_Management :  17
__label__Systemic_Risk_Management :  18
__label__Waste_And_Hazardous_Materials_Management :  19
__label__Water_And_Wastewater_Management :  20
__label__Air_Quality :  21
__label__Customer_Privacy :  22
__label__Ecological_Impacts :  23
__label__Energy_Management :  24
__label__GHG_Emissions :  25
```
