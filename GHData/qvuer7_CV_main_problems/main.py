
from utils.vizualisation import *
from config import *
from utils.data_loading import *


train_df, test_df = get_dataframes()
train_t, test_t = get_detection_transforms()
md = DetectionDataset(train_df, LABELS_MAP, transform = train_t)


image, target = md[200]

out = inference_faster_rcnn(checkpoint_path=CHECKPOINT_FOR_INFERENCE, image_ori = image)
print(out)



'''
if TASK==1:
    model, params = get_deep_lab_v3()
    train_dl, test_dl = get_segmentation_data_loaders()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params, lr=LR)
    train_segmentation(model = model, optimizer=optimizer, trainDataLoader= train_dl, testDataLoader = test_dl, criterion = criterion)

if TASK == 2:
    model, params = get_faster_rcnn()
    train_dl, test_dl = get_detection_data_loaders()
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    train_detector(model = model, optimizer = optimizer, trainLoader = train_dl, testLoader = test_dl)

if TASK == 3:
    model, params = get_mask_rcnn()
    train_dl, test_dl = get_instance_segmentation_data_loaders()
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    train_instance(model=model, trainLoader=train_dl, testLoader=test_dl, optimizer=optimizer)


'''