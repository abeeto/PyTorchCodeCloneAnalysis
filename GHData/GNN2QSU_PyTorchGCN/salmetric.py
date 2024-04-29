import os
from PIL import Image
from alisuretool.Tools import Tools


def get_file(mask_path, result_path):
    result_files = []
    for mask_name in os.listdir(mask_path):
        a = os.path.join(mask_path, mask_name)
        b = os.path.join(result_path, mask_name)
        if os.path.exists(a) and os.path.exists(b):
            if Image.open(a).size == Image.open(b).size:
                result_files.append("{} {}".format(b, a))
            else:
                Tools.print("{} {}".format(a, b))
        pass
    return result_files


"""
./salmetric salmetric.txt 36
"""


"""
mask_path = "/mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Mask"
result_path = "/mnt/4T/ALISURE/GCN/PyTorchGCN_Result/PYG_ChangeGCN_GCNAtt_NoAddGCN_NoAttRes/DUTS-TE/SOD"

mask_path = "/media/ubuntu/data1/ALISURE/DUTS/DUTS-TE/DUTS-TE-Mask"
result_path = "/media/ubuntu/data1/ALISURE/PyTorchGCN_Result/PYG_GCNAtt_NoAddGCN_NoAttRes/DUTS-TE/SOD"

Max F-measre: 0.894474
Precision:    0.920642
Recall:       0.817062
MAE:          0.0368984
mask_path = "/media/ubuntu/data1/ALISURE/DUTS/DUTS-TE/DUTS-TE-Mask"
result_path = "/media/ubuntu/data1/ALISURE/PyTorchGCN_Result/PYG_GCNAtt_NoAddGCN_NoAttRes_NewPool/DUTS-TE/SOD"


Max F-measre: 0.894308
Precision:    0.916229
Recall:       0.828253
MAE:          0.0370773
mask_path = "/media/ubuntu/data1/ALISURE/DUTS/DUTS-TE/DUTS-TE-Mask"
result_path = "/media/ubuntu/data1/ALISURE/PyTorchGCN_Result/PYG_GCNAtt_NoAddGCN_NoAttRes_Sigmoid/DUTS-TE/SOD"
"""


if __name__ == '__main__':
    mask_path = "/media/ubuntu/data1/ALISURE/DUTS/DUTS-TE/DUTS-TE-Mask"
    result_path = "/media/ubuntu/data1/ALISURE/PyTorchGCN_Result/PYG_GCNAtt_NoAddGCN_NoAttRes_Sigmoid/DUTS-TE/SOD"

    _result_files = get_file(mask_path, result_path)
    _txt = "\n".join(_result_files)
    Tools.write_to_txt("salmetric.txt", _txt, reset=True)
    pass
