import hnswlib
import numpy as np
from utils import DataCompiler
from nlp.inferences import create_infersent_model, create_fse_model
from nlp.embedding import create_fasttext_embeddings
from config import Config
from pathlib import Path

conf = Config()

print("Compiling data from csv files")
data = DataCompiler(["data.csv","data1.csv","data2.csv","data3.csv"])
data = data.compile_to_preprocessed_data()
print("Length of positions: ", len(data))

print("Creating FastText vectors")
create_fasttext_embeddings(data)

print("Creating InferSent model for indexes")
infsent = create_infersent_model(data)

# print("Creating SIF FSE model for indexes")
# fse, idxs = create_fse_model(data)


print("Start creating indexes")
dim = 4096
num_elements = len(data)

embeddings = []
labels = []
for i, sentence in enumerate(data):
    print(f"{i+1} of {len(data)} is embedding")
    # embeddings.append(fse.sv.get_vector(sentence["index"]))
    # labels.append(sentence["index"])
    embeddings.append(infsent.encode([" ".join(sentence["sentence"])])[0])
    labels.append(sentence["index"])

p = hnswlib.Index(space = 'l2', dim = dim)
p.init_index(max_elements = num_elements, ef_construction = 330, M = 28)
p.add_items(embeddings, labels)
p.set_ef(180)

labels, distances = p.knn_query(embeddings, k=1)
print("Recall for the positions:", np.mean(labels.reshape(-1) == np.arange(len(embeddings))))

print(f"Saving indexes to {conf.hnsw_indexes}")
Path(conf.hnsw_indexes.split("/")[0]).mkdir(parents=True, exist_ok=True)
p.save_index(conf.hnsw_indexes)