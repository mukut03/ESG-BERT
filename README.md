# ESG-BERT
**Domain Specific BERT Model for Text Mining in Sustainable Investing** 

Read more about this pre-trained model [here.](https://towardsdatascience.com/nlp-meets-sustainable-investing-d0542b3c264b?source=friends_link&sk=1f7e6641c3378aaff319a81decf387bf)

**In collaboration with [Charan Pothireddi](https://www.linkedin.com/in/sree-charan-pothireddi-6a0a3587/) and [Parabole.ai](https://www.linkedin.com/in/sree-charan-pothireddi-6a0a3587/)**

## Prerequisites
The further pre-trained ESG-BERT model can be found [here](https://drive.google.com/drive/folders/1yfNpMvByz3fJMsOqir3SerS6PwsRS2rt?usp=sharing) at this GitHub repository. It is a PyTorch model but it can be converted into a Tensorflow model. They can be fine-tuned using either framework. I found the PyTorch framework to be a lot cleaner, and easier to replicate with other models. However, serving the final fine-tuned model is a lot easier on TensorFlow, than on PyTorch. 

You can download the ESG-BERT model (named `pytorch_model.bin`) along with `config.json` and `vocab.txt` files here. The BERT base model was further pre-trained on Sustainable Investing text corpus, resulting in a domain specific model. You need the all of those 3 files for fine-tuning.  

The fine-tuned model for text classification is also available [here](https://drive.google.com/drive/folders/1Qz4HP3xkjLfJ6DGCFNeJ7GmcPq65_HVe?usp=sharing). It can be used directly to make predictions using just a few steps. 
First, download the fine-tuned `pytorch_model.bin`, `config.json`, and `vocab.txt` into your local directory. Make sure to place all of them into the same directory, mine is called `bert_model`.

### Install dependencies
JDK 11 is needed to serve the model. Go ahead and install it from the Oracle downloads page. Now we are ready to set up TorcheServe.
TorchServe is a model serving architecture for PyTorch models, go ahead and install that using pip. You can also use conda for the installation. We also need pytorch and transformers installed. 
``` bash
pip install torchserve torch-model-archiver
pip install torchvision 
pip install transformers
```

### Set up the handler script
Next up, we'll set up the handler script. It is a basic handler for text classification that can be improved upon. Save this script as `handler.py` in your directory. [1]
``` python
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
## Creating a torchserve model archive
Create a new model directory:
``` bash
mkdir model_store
```
TorcheServe uses a format called `MAR` (Model Archive). We can convert our PyTorch model to a `.mar` file using this command:
``` bash
torch-model-archiver --model-name "bert" --version 1.0 --serialized-file ./bert_model/pytorch_model.bin --extra-files "./bert_model/config.json,./bert_model/vocab.txt,./bert_model/index_to_name.json" --handler "./handler.py" --export-path "model_store/"
```
The resulting mar file will be stored in the `model_store` directory we just created.

## Serve the model
Finally, we can start TorchServe using the command: 
```
torchserve --start --model-store model_store --models bert=bert.mar
```

## Test the model
We can now query the model from another terminal window using the Inference API. We pass a text file containing text that the model will try to classify. 

```
curl -X POST http://127.0.0.1:8080/predictions/bert -T predict.txt
```
This returns a textual label, defined in the `index_to_name.json` file. 

## Fine-tuning the model yourself
For fine-tuning the model, you can use this command to load it into PyTorch. 
``` python
model = BertForSequenceClassification.from_pretrained(
    'path/to/dir/containing/ESG-BERT', 
    num_labels = num, #number of classifications
   output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.to(device)
```


References:
[1] - ---

 https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18


---
