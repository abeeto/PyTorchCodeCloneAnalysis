import argparse
import logging
import pprint
import os
import sys
import torch
from configurations import get_conf, Vocab

# from NER.decode import Evaluator
# from NER.dataset import get_ner_stream, get_text_stream
import preproc
from preproc.document import ParsedDocument
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
def do_ner(infiles, outdir, config, models):
    processor= NEREvaluator(config, models)
    for fname in infiles:
        doc = ParsedDocument()
        doc.load(fname)
        # processor.process(doc)
        processor.batch_process(doc)
        extract_tag_mentions(doc)
        bname= os.path.basename(fname)
        ofile= os.path.join(outdir, bname)
        doc.dump(ofile)
        print '{} processed'.format(doc._name)

# def do_ner_encdec(infiles, outdir, config, models):
#     processor= EncDecJsonEvaluator(config, models)
#     for fname in infiles:
#         doc = ParsedDocument()
#         doc.load(fname)
#         processor.process(doc)
#         extract_tag_mentions(doc)
#         bname= os.path.basename(fname)
#         ofile= os.path.join(outdir, bname)
#         doc.dump(ofile)
#         print '{} processed'.format(doc._name)
# def do_coref(infiles, outdir, config, models):
#     processor= CorefEvaluator(config, models)
#     for fname in infiles:
#         doc = ParsedDocument()
#         doc.load(fname)
#         processor.process(doc)
#
#         bname= os.path.basename(fname)
#         ofile= os.path.join(outdir, bname)
#         doc.dump(ofile)
#         print '{} processed'.format(doc._name)
        

    

if __name__ == "__main__":
    # Get configurations for model
    # config = getattr(configurations, args.proto)()
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_arg', help='mode', type=int)
    parser.add_argument('--lang', help='lang')
    args = parser.parse_args()
    my_arg = args.my_arg
    # config = get_conf(args.lang)
    config = get_conf('cmn')
    config['model_dir'] = 'cmn_bio_model1'
    models = None
    in_dir = '/home/ryk/programming/kbp_src/ner/datagen/train/cmn_json/2017/eval/0'#+str(my_arg)
    json_files= preproc._list_files(in_dir, '.json')
    # out_dir = args.out_dir
    out_dir = '/home/ryk/programming/kbp_src/ner/datagen/train/2016_model_json'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # task_name = args.task
    task_name = 'ner'
    if task_name=='ner':
        do_ner(json_files, out_dir, config, models)
    # elif task_name=='ner_encdec':
    #     do_ner_encdec(json_files, out_dir, config, models)
    # elif task_name=='coref':
    #     do_coref(json_files, out_dir, config, models)
    else:
        print 'unknown task {}'.format(task_name)
    
    
    
    
