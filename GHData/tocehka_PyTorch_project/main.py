from utils import DataCompiler
from utils.dlap import dirty_load_and_process, benchmark
from nlp.inferences import create_infersent_model, create_fse_model
from search_engine.core import SearchPosition
from nlp.preprocessing.utils import prepare_sentence
import numpy as np

@benchmark
def collate_DB_bench(clean, dirty, model, engine):
    search = SearchPosition(model, engine=engine)
    labels, distances = search.search(dirty, n=7, logs=False)
    return labels, distances

def collate_DB(dirty, model, engine):
    search = SearchPosition(model, engine=engine)
    labels, distances = search.search(dirty, n=5, logs=False)
    return labels, distances

if __name__ == "__main__":
    print("Compiling data from csv files")
    data = DataCompiler(["data.csv","data1.csv","data2.csv","data3.csv"])
    data = data.compile_to_preprocessed_data()
    num_elements = len(data)
    print("Number of positions:", num_elements)

    infsent_model = create_infersent_model(data)
    #fse_model, indxs = create_fse_model(data)

    dirty_data = dirty_load_and_process()
    #print(dirty_data)

    #labels, distances = collate_DB_bench(data, dirty_data, fse_model, "fse")
    labels, distances = collate_DB_bench(data, dirty_data, infsent_model, "hnsw")
    #labels, distances = collate_DB_bench(data, dirty_data, fse_model, "fse_hnsw")

    """ query = prepare_sentence(["смеситель l4299a Ledeme"])
    print("Search for:", query)
    labels, distances = collate_DB(query, fse_model, "fse")
    labels, distances = labels[0], distances[0]
    print("FSE:")
    for i in range(len(labels)):
        print(i, ":", data[labels[i]], distances[i])

    labels, distances = collate_DB(query, infsent_model, "hnsw")
    print("HNSW:")
    labels, distances = labels[0], distances[0]
    for i in range(len(labels)):
        print(i, ":", data[labels[i]], distances[i]) """
    