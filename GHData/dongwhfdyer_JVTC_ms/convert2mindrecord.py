import mindspore.ops as ops
from io import BytesIO
import os
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.c_transforms as vision
from PIL import Image


def save_mind_record(dataset_dir, txt_path, mindrecord_path, transform_state='train', num_cam=8, K=4):
    if os.path.exists(mindrecord_path):
        os.remove(mindrecord_path)
        os.remove(mindrecord_path + ".db")

    writer = FileWriter(file_name=mindrecord_path, shard_num=1)

    cv_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema, "it is a cv dataset")

    writer.add_index(["file_name", "label"])

    data = []
    with open(txt_path) as f:
        line = f.readlines()
        img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
        label_list = [int(i.split()[1]) for i in line]
        cam_list = [int(i.split()[2]) for i in line]
    for i in range(len(img_list)):
        sample = {}
        write_io = BytesIO()
        img_path = img_list[i]
        camid = cam_list[i]

        randperm = ops.Randperm(num_cam)
        cams = randperm(num_cam) + 1

        imgs = []
        for sel_cam in cams[0:K]:

            if sel_cam != camid:
                im_path_cam = img_path[:-4] + '_fake_' + str(camid) + 'to' + str(sel_cam.numpy()) + '.jpg'
            else:
                im_path_cam = img_path
            imgs.append(im_path_cam)


        Image.open(im_path_cam).save(write_io, format='JPEG')
        sample["file_name"] = img_path
        sample["label"] = label_list[i]
        sample["data"] = write_io.getvalue()

        data.append(sample)
        if i % 10 == 0:
            writer.write_raw_data(data)
            data = []
    if data:
        writer.write_raw_data(data)
    writer.commit()

    # data = []
    # for i in range(100):
    #     i += 1
    #     sample = {}
    #     white_io = BytesIO()
    #     Image.new('RGB', (i * 10, i * 10), (255, 255, 255)).save(white_io, 'JPEG')
    #     image_bytes = white_io.getvalue()
    #     sample['file_name'] = str(i) + ".jpg"
    #     sample['label'] = i
    #     sample['data'] = white_io.getvalue()
    #     data.append(sample)
    #     if i % 10 == 0:
    #         writer.write_raw_data(data)
    #         data = []
    # if data:
    #     writer.write_raw_data(data)
    # writer.commit()


if __name__ == '__main__':
    MINDRECORD_FILE = "test.mindrecord"
    dataset_dir = r""
    txt_path = r""

    save_mind_record(dataset_dir, txt_path, MINDRECORD_FILE)

    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE, )