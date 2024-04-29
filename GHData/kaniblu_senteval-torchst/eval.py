import os
import yaml
import datetime
import logging
import subprocess

import yaap
import senteval
import numpy as np


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    return


def batcher(params, samples):

    #if type(samples) != list:
    #    print("samples is not a list")
    #    print(samples, type(samples))
    #elif type(samples[0]) != list:
    #    print("the first element of samples is not a list")
    #    print(samples[0], type(samples[0]))
    #elif len(samples[0]) > 0 and type(samples[0][0]) != str:
    #    print("the first element of the first element of samples is not str")
    #    print(samples[0][0], type(samples[0][0]))
    #    print(samples[0])

    samples = [" ".join(x.decode("utf-8") if type(x) == bytes else x for x in s) for s in samples]
    print("[{}] n={}, ex={}".format(datetime.datetime.now(), len(samples), samples[0] if samples else ""))
    model = params.model
    data = model(samples)

    return data


class SentenceVectorizer(object):
    def __init__(self, config_path, package_name="torchst"):
        self.config_path = config_path
        self.package_name = package_name
        self.process = subprocess.Popen(["python","-m", "{}.vectorize".format(package_name),
                                         "--config", config_path], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def __call__(self, sents):
        # print([len(x) for x in sents])
        encoded = ("\n".join(sents) + "\n").encode("utf-8")
        self.process.stdin.write(encoded)
        self.process.stdin.flush()
        res = []

        for _ in range(len(sents)):
            res.append(self.process.stdout.readline())

        data = [np.fromstring(r, dtype=float, sep=' ') for r in res]
        assert len(sents) == len(data)
        # print([(s, x) for s, x in zip(sents, data)])
        data = np.vstack(data)

        return data


class dotdict(dict):
    __getattr__ = dict.get
    __selattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    parser = yaap.ArgParser(allow_config=True)
    parser.add("--vectorizer-config", type=yaap.path, required=True)
    parser.add("--senteval-data", type=yaap.path, required=True)
    parser.add("--tasks", type=str, action="append", required=True)
    parser.add("--batch-size", type=int, default=None)

    args = parser.parse_args()

    assert os.path.exists(args.vectorizer_config)

    with open(args.vectorizer_config, "r") as f:
        vec_conf = yaml.load(f)

    if args.batch_size is None:
        assert "batch-size" in vec_conf

        batch_size = vec_conf.get("batch-size")
    else:
        batch_size = args.batch_size

    sv = SentenceVectorizer(args.vectorizer_config)

    params = {
        "usepytorch": True,
        "task_path": args.senteval_data,
        "batch_size": batch_size,
        "model": sv,
    }

    se = senteval.SentEval(dotdict(params), batcher, prepare)
    se.eval(args.tasks)


if __name__ == "__main__":
    main()
