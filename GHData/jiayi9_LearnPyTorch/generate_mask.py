import argparse
import base64
import json
import os
import os.path as osp
import io
import numpy as np
from tqdm import tqdm
import imgviz
import PIL.Image
import yaml

# image to array
def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

# Convert labels shape to array
def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = instance_names.index(label)
        cls_id = label_name_to_value[cls_name]
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls

# Convert shape to mask
def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

# Save mask to image
def lblsave(filename, lbl):
    import imgviz

    if osp.splitext(filename)[1] != '.png':
        filename += '.png'
    print(filename)
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )


# Convert shape to mask
def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


# Save mask to image
def lblsave(filename, lbl):
    import imgviz

    if osp.splitext(filename)[1] != '.png':
        filename += '.png'
    #     print(filename)
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )


def main():
    # Receive parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    # json_file and output folder name
    json_file = args.json_file

    # Generate folder from json_file 
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    # Load json
    data = json.load(open(json_file))
    imageData = data.get('imageData')

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = img_b64_to_arr(imageData)
    # print("The dimension of image is {}".format(img.ndim))

    # Generagte labels
    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    # 
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    # RGB or Gray images
    if img.ndim == 3:
        lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.rgb2gray(img), label_names=label_names, loc='rb')
    elif img.ndim == 2:
        lbl_viz = imgviz.label2rgb(
        label=lbl, img=img, label_names=label_names, loc='rb')

    # Visualize label images
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    args.json_file.split('.')[0]
    print(out_dir)
    lblsave(osp.join(out_dir, args.json_file.split('.')[0] + '_label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

#if __name__ == '__main__':
#    main()


# Convert json file to several images
def json_to_image(json_file, out_dir, viz_image = True, label_image = False):
    # Load json
    data = json.load(open(json_file))
    imageData = data.get('imageData')

    # Convert image to array
    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = img_b64_to_arr(imageData)
    # print("The dimension of image is {}".format(img.ndim))

    # Generate labels
    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    # Generate label names
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    # RGB or Gray images
    if img.ndim == 3:
        lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.rgb2gray(img), label_names=label_names, loc='rb')
    elif img.ndim == 2:
        lbl_viz = imgviz.label2rgb(
        label=lbl, img=img, label_names=label_names, loc='rb')

    # Save raw image from Json structure
    # PIL.Image.fromarray(img).save(osp.join(out_dir, json_file.split('\\')[-1].split('.')[0] + '_img.png'))
    # PIL.Image.fromarray(img).save(osp.join(out_dir, json_file.split('\\')[-1].split('.')[0] + '_img.bmp'))

    # Save mask
    if label_image:
#        print(out_dir)
#        print(json_file)
#        print(json_file.split('\\')[-1].split('.')[0])
#        print(json_file.split('\\'))
#        print('\n')
#        print(out_dir + json_file.split('\\')[-1].split('.')[0] + '_mask.png')
#        PATH = osp.join(out_dir, json_file.split('\\')[-1].split('.')[0] + '_mask.png')
#        print(PATH)
        
        # Premise, / is use,   .json is the ext name
        filename = json_file.split('/')[-1]
        adjusted_filename = filename[0:-5]
        PATH = out_dir + adjusted_filename + '_mask.png'

        lblsave(PATH, lbl)
    
    # Save heatmap
    if viz_image:
        path = osp.join(out_dir, json_file.split('\\')[-1].split('.')[0] + '_viz.png')
        PIL.Image.fromarray(lbl_viz).save(path)


#json_file = "C:/PyTorch/lujiayi/code_decode/json_2/376200703000675_2_0.496_1593744848.bmp"
#filename = x.split('/')[-1]




# Generate viz and label files
# json_folder = 'C:/PyTorch/lujiayi/code_decode/img_and_json/'
# out_dir = 'C:/PyTorch/lujiayi/code_decode/generated_masks/'
# Convert json to image 

# for json_file in tqdm(os.listdir(json_folder)):
#     print(os.path.join(json_folder, json_file))
#     if 'json' in json_file:
#         json_to_image(json_folder + json_file, out_dir, viz_image = False, label_image = True)    
# print("Conversion is Complete!")


def folder_json_to_mask_png(json_folder, out_dir):
    N = len(os.listdir(json_folder))
    N2 = len(os.listdir(out_dir))
    print('Json folder:', N, '  Output folder:', N2)
    for json_file in tqdm(os.listdir(json_folder)):
        print(os.path.join(json_folder, json_file))
        if 'json' in json_file:
            json_to_image(json_folder + json_file, out_dir, viz_image = False, label_image = True)    
    N = len(os.listdir(json_folder))
    N2 = len(os.listdir(out_dir))
    print('Json folder:', N, '  Output folder:', N2)
    if N == N2:
        print('Mask generation done (file quantity same)')
        print('')
        return True
    else:
        print('Mask generation problematic (json file quantity !=  png file quantity)')
        print('')
        return False
        


## Covert label_image to csv
#LABEL_PATH = 'c:\\Users\\YAK6SGH\\Desktop\\label\\'
#for label_file in os.listdir(LABEL_PATH):
#    label_png = LABEL_PATH + label_file
#    np.set_printoptions(threshold=np.inf)  
#    lbl = np.asarray(PIL.Image.open(label_png))
#    df = pd.DataFrame(lbl)
#    df.to_csv(LABEL_PATH + label_file.split('.')[0] + '.csv')
    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_folder', type = str, default = 'none')
parser.add_argument('--out_dir', type = str, default = 'none')

args = parser.parse_args()

print('Model:', args.json_folder)
print('Model name:', args.out_dir)


folder_json_to_mask_png(args.json_folder, args.out_dir)
