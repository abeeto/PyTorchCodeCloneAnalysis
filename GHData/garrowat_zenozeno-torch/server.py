# Flask stuff
from flask import Flask, url_for, make_response, request
from flask_classful import FlaskView, route
from flask_cors import CORS, cross_origin

import json

from transformers import GPT2LMHeadModel, AutoTokenizer

app = Flask(__name__)
CORS(app)

def output_json(data, code, headers=None):
    content_type = 'application/json'
    dumped = json.dumps(data)
    if headers:
        headers.update({'Content-Type': content_type})
    else:
        headers = {'Content-Type': content_type}
    response = make_response(dumped, code, headers)
    return response

tokenizer = AutoTokenizer.from_pretrained("gpt2", eos_token=".")
model = GPT2LMHeadModel.from_pretrained("./models", cache_dir=None, from_tf=False, state_dict=None)

def generateQuotes(
  input="The meaning of life is",
  num_quotes=1,
  max_length=100,
  top_k=40,
  top_p=0.9,
  temperature=0.7
):
  input = tokenizer.encode(input, return_tensors="pt")
  if num_quotes > 5:
    num_quotes = 5

  outputs = model.generate(
    input,
    do_sample=True,
    max_length=max_length,
    top_k=top_k,
    top_p=top_p,
    temperature=temperature,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=num_quotes,
  )
  return outputs;

class TestView(FlaskView):
  route_base = '/zenozeno'
  representations = {'application/json': output_json}

  @route('/test/<input>')
  def get_test(self, input):
    return {"test": input}

class QuotesView(FlaskView):
  route_base = '/zenozeno/quotes'
  representations = {'application/json': output_json}

  @app.route('/')
  def post(self):
    data = request.get_json()
    quotes = []
    for sample_output in generateQuotes(
        data['input'],
        data['numberOfQuotes'],
        data['maxQuoteLength'],
        data['topK'],
        data['topP'],
        data['temperature']
      ):
      quotes.append(tokenizer.decode(sample_output, skip_special_tokens=True))
    return {"quotes": quotes}

TestView.register(app)
QuotesView.register(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
