import argparse
import pandas as pd
import numpy as np

import torch
from torchfm.model.afn import AdaptiveFactorizationNetwork

        
feature_cols = ['cnt_date', 'geo_tokyo', 'geo_osaka', 'width', 'height',
                'age_range_undetermined', 'age_range_18_24', 'age_range_25_34', 'age_range_35_44',
                'age_range_45_54', 'age_range_55_64', 'age_range_65_more', 'gender_male', 'gender_female',
                'list_type_rule_based', 'list_type_logical', 'list_type_remarketing', 'list_type_similar',
                'list_type_crm_based', 'ad_type', 'status', 'device']


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', type=str, default='./test.csv')
    parser.add_argument('--save_path', type=str, default='result.csv')
    parser.add_argument('--load_path', type=str, default='./model_inside.pth')
    
    args = parser.parse_args()

    return args


def unscale(output):

    min_num = -7.600902459542082
    max_num = 4.605175185975591

    output = output*(max_num - min_num)    
    output = np.exp(output + min_num)

    return output - 0.0005


def test_model(model, loader, device):

    model.to(device)
    model.eval()
    
    #out_csv = pd.DataFrame(columns=["ids", "preds"])

    row_list = []
    
    for i, (fields, idx) in enumerate(loader):

        dict_row = {}
        
        fields = fields.to(device)
        with torch.no_grad():
            y = model(fields)

            y = y.squeeze(0)
            ret = y.cpu().numpy()

        ret = unscale(ret)
        dict_row["ids"] = idx
        dict_row["preds"] = ret

        row_list.append(dict_row)
        
    out_csv = pd.DataFrame(row_list)
    return out_csv
    
            
def to_tensor(np_array):

    tensor = torch.from_numpy(np_array)
    return tensor.unsqueeze(0)

        
def load_data(pd_path):

    pd_data = pd.read_csv(pd_path)
    
    for i, row in pd_data.iterrows():
        
        feature = row[feature_cols].to_list()
        feature = np.array(feature)
        
        feature = to_tensor(feature)
        index = row["idx"]
        
        yield feature, index

        
if __name__ == '__main__':

    args = get_args()
    loader = load_data(args.input_csv)
    
    field_dims = [49, 50, 50, 7, 8, 20, 20, 20, 20, 20, 20, 20, 50, 50, 20, 20, 10, 10, 5, 3, 3, 5]
    model = AdaptiveFactorizationNetwork(field_dims=field_dims, embed_dim=16, LNN_dim=1500,
                                         mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    
    model.load_state_dict(torch.load(args.load_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    out_csv = test_model(model, loader, device)
    out_csv.to_csv(args.save_path, index=False)

    
