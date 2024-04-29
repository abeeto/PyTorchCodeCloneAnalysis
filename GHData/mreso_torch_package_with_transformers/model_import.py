
from torch import package


package_path = 'model.pt'
bert_package_imp = package.PackageImporter(package_path)

pickled_model = bert_package_imp.load_pickle("model", "model.pkl")

# print('TorchPackaged Model')
# print(run_model(pickled_model, sequence_0, sequence_1))
# print(run_model(pickled_model, sequence_0, sequence_2))