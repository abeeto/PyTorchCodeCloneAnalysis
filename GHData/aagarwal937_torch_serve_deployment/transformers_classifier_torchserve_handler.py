
"""
Handler for semantic_search using SentenceTransformer 
"""


# importing standard modules ==================================================
from typing import Dict
import json


# importing third-party modules ===============================================
import numpy
from sentence_transformers import SentenceTransformer


# class definitions ===========================================================
class NumpyArrayEncoder(json.JSONEncoder):
    r""" class inheriting from 'json.JSONEncoder' and overriding the 'default'
    instance method. """


    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


    pass # end of NumpyArrayEncoder


class SematicSearch(object):
    """
    SematicSearch handler class. This handler takes a corpus and query strings
    as input and returns the closest 5 sentences of the corpus for each query 
    sentence based on cosine similarity.
    """


    def __init__(self):
        super(SematicSearch, self).__init__()
        self.initialized: bool = False
        self.embedder: SentenceTransformer = None

        return
    

    def initialize(self, context):  # some Context object

        properties: Dict = context.system_properties
        model_dir = properties.get("model_dir")
        print(model_dir)        
        self.embedder = SentenceTransformer(model_dir)
        self.initialized = True

        return

    
    def preprocess(self, data):

        print(data)
        inputs = data[0].get("data")
        print(inputs)
        if inputs is None:
            inputs = data[0].get("body")
        inputs = inputs.decode('utf-8')
        inputs = json.loads(inputs)
        queries = inputs['queries']

        return queries
    

    def inference(self, data):
        query_embeddings = self.embedder.encode(data)
        return query_embeddings


    def postprocess(self, data):
        return [json.dumps(data, cls=NumpyArrayEncoder)]

    
    pass # end of SematicSearch


_service = SematicSearch()


# main entry point of the module ==============================================
def handle(data, context):
    """
    Entry point for SematicSearch handler
    """
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
        raise Exception("Unable to process input data. " + str(e))


# Tester
"""class Ctx(object):
    pass
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
data = {'corpus':corpus,'queries':queries}
properties = {}
properties["model_dir"] = '/Users/dhaniram_kshirsagar/projects/neo-sagemaker/mms/code/serve/examples/semantic_search'
ctx = Ctx( )
ctx.system_properties = properties
output = handle([{'data':json.dumps(data)}],ctx)
print(output)"""