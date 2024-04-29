import torch

# In case you got PosixPth problem on windows, add these lines
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_learner(file_path, file='export.pkl', is_fastaiv2=False, test=None, tfm_y=None, **db_kwargs):
    model = []
    mean = []
    std = []
    """Load a `Learner` object saved with `export_state` in `path/file` with empty data, 
    optionally add `test` and load on `cpu`. `file` can be file-like (file or buffer) - FastAI"""

    source = file_path + '/' + file
    state = torch.load(source, map_location='cpu') if not torch.cuda.is_available() else torch.load(source)
    if is_fastaiv2:
        model = state.model
    else:  # FastAI_v1
        model = state.pop('model')
        try:
            mean = state['data']['normalize']['mean']
            std = state['data']['normalize']['std']
        except Exception as e:
            print(e)
        # model.load_state_dict(state, strict=True)
    return model, mean, std


modelpath = r'E:\PythonScripts\ModelConversionTest'
modelfile = 'yourFastAI_model.pkl'

model, mean, std = load_learner(modelpath, modelfile, is_fastaiv2=True)

model.eval()

# This is important, check which size of image is needed for your model. Usually, it will be good to use what you used on training
# x = torch.randn(1, 3, [height], [width], requires_grad=False)
x = torch.randn(1, 3, 240, 320, requires_grad=False)  #.cuda()
torch_out = torch.onnx._export(model, x, r'E:\PythonScripts\ModelConversionTest\ONNXModel.onnx', export_params=True)

with open("model_data_mean_std.json", "w") as f:
    txt_lines = f'"mean": {mean}, "std": {std}'
    f.write(txt_lines)
print("DONE")
