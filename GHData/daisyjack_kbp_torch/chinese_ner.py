# coding: utf-8

import argparse
import logging
import pprint
import os
import sys
import torch
from configurations import get_conf, Vocab
import json

# from NER.decode import Evaluator
# from NER.dataset import get_ner_stream, get_text_stream
import preproc
from preproc.document import ParsedDocument, TextDocument
from preproc.corenlp import StanfordCoreNLP
# from ner_eval.json_data import extract_tag_mentions,CorefEvaluator, NEREvaluator
from ner_eval.json_data import extract_tag_mentions, NEREvaluator

# from ner_eval.json_proc import EncDecJsonEvaluator
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s[%(levelname)s] %(name)s  - %(message)s',
                    )

logger = logging.getLogger(__name__)


# Get the arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--proto",  default="get_conf_ner",
#                     help="Prototype config to use for config")
# parser.add_argument('--task', default= 'ner',help='ner|coref')
# parser.add_argument('models', help = 'model files, split by \":\"')
# parser.add_argument('in_dir', help = 'path for input files')
# parser.add_argument('out_dir', help = 'path for output files')
# args = parser.parse_args()
def do_ner(doc, config, models):
    processor = NEREvaluator(config, models)
    processor.batch_process(doc)

def text2json(text, lang):
    corenlp = StanfordCoreNLP(lang=lang, server_url='http://127.0.0.1:9000')
    doc = TextDocument(text)
    doc.parse_document(corenlp)
    doc.split_sentence()
    return doc


def get_cmn_mention(text):
    """
    Returns a string of json format

    Args:
        text  (str): must be of the format of utf-8

    """
    doc = text2json(text, 'cmn')
    config = get_conf('cmn')
    config['model_dir'] = 'cmn_bio_model1'
    models = None
    do_ner(doc, config, models)
    result = {'annotate': doc._annotate, 'origin_text': doc._origin_text, 'text_spans': doc.text_spans}
    pretty_result = {'annotate': [], 'origin_text': result['origin_text']}
    for ann in result['annotate']:
        tag = ann['md_tag']
        start_token = ann['mention_tokens'][0].split('_')
        end_token = ann['mention_tokens'][-1].split('_')
        sent = int(start_token[0])
        start = result['text_spans'][sent]['tokens'][int(start_token[1])]['char_begin']
        end = result['text_spans'][sent]['tokens'][int(end_token[1])]['char_end']
        mention_text = result['origin_text'][start:end]
        mention = {'char_begin': start, 'char_end': end, 'text': mention_text, 'tag': tag}
        pretty_result['annotate'].append(mention)

    return json.dumps(pretty_result)


    # return json.dumps(result)



if __name__ == "__main__":
    text = "【军情速递】从数据看安倍经济学 好象失败了。" \
           "美国是主使，日本是仆从，任何人也改变不了它们这种关系，没有美国的鼓动，要不是当初美国将钓鱼岛的所谓“管辖权”交给日本，日本也不敢这么嚣张！" \
           "因此打起仗来就必须把美日及其帮凶一起考虑，否则，那可就是世界级傻瓜了！" \
           "安倍的国际地位臭名远扬。".decode('utf-8')
    result = get_cmn_mention(text)
    print result






