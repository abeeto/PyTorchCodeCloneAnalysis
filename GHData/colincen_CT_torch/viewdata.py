from model.concept_tagger import ConceptTagger
from tools.DataSet.SNIPS import snips
from tools.utils import *
import torch
import copy
target_domain = 'PlayMusic'
model_path = '/home/sh/code/CT_torch_no_neg/data/PlayMusic/model'
device = 'cpu'

slots = ['album', 'artist', 'best_rating', 'city', 'condition_description',
         'condition_temperature', 'country', 'cuisine', 'current_location', 'entity_name',
         'facility', 'genre', 'geographic_poi', 'location_name', 'movie_name', 'movie_type',
         'music_item', 'object_location_type', 'object_name', 'object_part_of_series_type', 'object_select',
         'object_type', 'party_size_description', 'party_size_number', 'playlist', 'playlist_owner', 'poi',
         'rating_unit', 'rating_value', 'restaurant_name', 'restaurant_type', 'served_dish', 'service',
         'sort', 'spatial_relation', 'state', 'timeRange', 'track', 'year']


model = ConceptTagger.load(model_path, device)
config = model.config
dataDict = getNERdata(dataSetName=config.dataset,
                          dataDir=config.data_dir,
                          desc_path=config.description_path,
                          cross_domain=config.cross_domain,
                          target_domain=config.target_domain)

# with open('rawtest.txt', 'w') as f:
#     for i in dataDict['target']['test']:
#         f.write(str(i['tokens'])+'\n')
#         f.write(str(i['NER_BIO'])+'\n')
#         f.write(str(i['slot'])+'\n')
#         f.write('\n')
# f.close()

temple = [[['play', 'the', 'song', 'i', 'get', 'ideas', 'as', 'performed', 'by', 'richard', 'kruspe'],
        ['O', 'B-entity_name', 'I-entity_name', 'O', 'B-playlist', 'I-playlist', 'I-playlist', 'I-playlist', 'I-playlist', 'O'],
        ['playlist']]]

data = []
for i in range(len(slots)):
    temp = []
    temp.append(copy.deepcopy(temple[0][0]))
    temp.append(copy.deepcopy(temple[0][1]))
    temp.append(copy.deepcopy(slots[i]))
    data.append(temp)



model.to(device)
model.eval()
with torch.no_grad():
    for pa in test_generator(data, 1):
        x = pa[0]
        y = pa[1]
        slot = pa[2]
        p = model.Eval(x, slot)[0]
        for k in range(len(p)):
            p[k] = p[k][0]
        print(slot)
        print(p)

# test_metric_pre, test_metric_rec, test_metric_f1 = evaluate(model, dataDict['target']['test'], config.batch_size, log)
# print("test_pre : %.4f, test_rec : %.4f, test_f1 : %.4f" % (test_metric_pre, test_metric_rec, test_metric_f1),
#       file=sys.stderr)

