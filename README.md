
```bash
pip install bert-extractive-summarizer
```

## Examples

### Simple Example
```python
from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer()
model(body)
model(body2)
```

#### Specifying number of sentences

Number of sentences can be supplied as a ratio or an integer. Examples are provided below.

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer()
result = model(body, ratio=0.2)  # Specified with ratio
result = model(body, num_sentences=3)  # Will return 3 sentences 
```

#### Using multiple hidden layers as the embedding output

You can also concat the summarizer embeddings for clustering. A simple example is below.

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
result = model(body, num_sentences=3)
```

```
pip install -U sentence-transformers
```

Then a simple example is the following:

```python
from summarizer.sbert import SBertSummarizer

body = 'Text body that you want to summarize with BERT'
model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
result = model(body, num_sentences=3)
```

It is worth noting that all the features that you can do with the main Summarizer class, you can also do with SBert.

### Retrieve Embeddings
You can also retrieve the embeddings of the summarization. Examples are below:

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer()
result = model.run_embeddings(body, ratio=0.2)  # Specified with ratio. 
result = model.run_embeddings(body, num_sentences=3)  # Will return (3, N) embedding numpy matrix.
result = model.run_embeddings(body, num_sentences=3, aggregate='mean')  # Will return Mean aggregate over embeddings. 
```

### Use Coreference
First ensure you have installed neuralcoref and spacy. It is worth noting that neuralcoref does not work with spacy > 0.2.1.
```bash
pip install spacy
pip install transformers # > 4.0.0
pip install neuralcoref

python -m spacy download en_core_web_md
```

Then to to use coreference, run the following:

```python
from summarizer import Summarizer
from summarizer.text_processors.coreference_handler import CoreferenceHandler

handler = CoreferenceHandler(greedyness=.4)
# How coreference works:
# >>>handler.process('''My sister has a dog. She loves him.''', min_length=2)
# ['My sister has a dog.', 'My sister loves a dog.']

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer(sentence_handler=handler)
model(body)
model(body2)
```

### Custom Model Example
```python
from transformers import *

from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
model(body)
model(body2)
```

### Large Example

```python
from summarizer import Summarizer

from summarizer import Summarizer

body = 'Your Text here.'
model = Summarizer()
res = model.calculate_elbow(body, k_max=10)
print(res)
```

You can also find the optimal number of sentences with elbow using the following algorithm.

```python
from summarizer import Summarizer

body = 'Your Text here.'
model = Summarizer()
res = model.calculate_optimal_k(body, k_max=10)
print(res)
```

## Summarizer Options

```
model = Summarizer(
    model: This gets used by the hugging face bert library to load the model, you can supply a custom trained model here
    custom_model: If you have a pre-trained model, you can add the model class here.
    custom_tokenizer:  If you have a custom tokenizer, you can add the tokenizer here.
    hidden: Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
    reduce_option: It can be 'mean', 'median', or 'max'. This reduces the embedding layer for pooling.
    sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass CoreferenceHandler instance
)

model(
    body: str # The string body that you want to summarize
    ratio: float # The ratio of sentences that you want for the final summary
    min_length: int # Parameter to specify to remove sentences that are less than 40 characters
    max_length: int # Parameter to specify to remove sentences greater than the max length,
    num_sentences: Number of sentences to use. Overrides ratio if supplied.
)
```

## Running the Service

There is a provided flask service and corresponding Dockerfile. Running the service is simple, and can be done though 
the Makefile with the two commands:

```
make docker-service-build
make docker-service-run
```

This will use the Bert-base-uncased model, which has a small representation. The docker run also accepts a variety of 
arguments for custom and different models. This can be done through a command such as:

```
docker build -t summary-service -f Dockerfile.service ./
docker run --rm -it -p 5000:5000 summary-service:latest -model bert-large-uncased
```

Other arguments can also be passed to the server. Below includes the list of available arguments.

* -greediness: Float parameter that determines how greedy nueralcoref should be
* -reduce: Determines the reduction statistic of the encoding layer (mean, median, max).
* -hidden: Determines the hidden layer to use for embeddings (default is -2)
* -port: Determines the port to use.
* -host: Determines the host to use.

Once the service is running, you can make a summarization command at the `http://localhost:5000/summarize` endpoint. 
This endpoint accepts a text/plain input which represents the text that you want to summarize. Parameters can also be 
passed as request arguments. The accepted arguments are:
