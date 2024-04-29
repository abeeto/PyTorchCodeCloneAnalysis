"""Load data from jsonlines files into an ElasticSearch index."""

import json
import time
from itertools import islice, chain
from typing import Text, List
from argparse import Namespace

import tqdm
import numpy as np
from sklearn import metrics
from elasticsearch import Elasticsearch

import torch

from models.citation_model import CitationNegModel, preprocess


ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'pubmed_articles'


def run_eval(client, k, page_size, fields: List[Text], rerank_model=None):
    ex_batch_iter = edge_iter(client, limit=1000, page_size=page_size)
    y_true = []
    y_pred = []
    total = 0
    progress = tqdm.tqdm(unit="example")
    for ex_batch in ex_batch_iter:
        model_out = model.predict(ex_batch)
        print(model_out)
        responses = msearch(client,
                            [ex['context'] for ex in ex_batch],
                            'pubmed_articles',
                            size=k,
                            fields=fields)['responses']
        progress.update(page_size)
        for i, response in enumerate(responses):
            results = response['hits']['hits']
            if not results:
                continue
            cited_pmid = ex_batch[i]['cited_pmid']
            context = ex_batch[i]['context']

            y_t = [int(result["_source"].get('pmid') == cited_pmid) for result in results]
            y_t += [0] * (k - len(y_t))
            if any(y_t):
                total += 1

            y_true.append(y_t)
            y_p = [result["_score"] for result in results]
            y_p += [0] * (k - len(y_p))
            y_pred.append(y_p)
            if i % (page_size * 5) == 0:
                progress.set_postfix_str(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
    print("\n\n")
    print(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
    print(f"A total of {total} correct results retrieved in top {k}.")


if __name__ == "__main__":

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    args = {
        'batch_size': 16,
        'lr': 0.0001,
        'b1': 0.9,
        'b2': 0.999,
        'freeze_layers': list(range(11)),
        'dropout': 0.4,
        'weight_decay': 0.15,
        'train_data': '/mnt/atlas/cit_pred_jsonlines/train.jsonl',
        'val_data': '/mnt/atlas/cit_pred_jsonlines/validate.jsonl'
    }
    model = CitationNegModel(Namespace(**args))
    model.load_state_dict(torch.load('/mnt/atlas/models/model.pt', map_location="cuda:0"))
    run_eval(client, k=200, page_size=50, fields=['abstract^2', 'context^4', 'text'], rerank_model=model)
