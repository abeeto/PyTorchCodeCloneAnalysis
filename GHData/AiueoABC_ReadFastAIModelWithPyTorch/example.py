import torch

# In case you got PosixPth problem on windows, add these lines
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_learner(file_path, file='export.pkl', is_fastaiv2=False):
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

if __name__=="__main":
    # Set file path and name
    filepath = 'path/to/your/FastAIModel/dir'
    filename = 'yourFastAIModel.pkl'
    # Say True if you use FastAI v2
    is_fastAIv2 = False
    print(load_learner(filepath, filename, is_fastAIv2))
