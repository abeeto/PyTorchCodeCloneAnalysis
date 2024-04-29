from commons import *

def box_resize(image, boxes, dims=(300,300), return_percent_coords=True):
    new_image = FT.resize(image, dims)
    new_boxes = boxes

    # no need to unsqueeze a dimension, this is already broadcastable, in this case
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height])
    new_boxes = new_boxes/old_dims # broadcasting

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]])
        new_boxes = new_boxes*new_dims # broadcasting

    return new_image, new_boxes

def photometric_distort(image):
    new_image = image

    distortions = [FT.adjust_brightness, FT.adjust_contrast, FT.adjust_saturation, FT.adjust_hue]
    random.shuffle(distortions)

    for d in distortions:
        if random.random()<0.5:
            if d.__name__ is 'adjust_hue':
                adjust_factor = random.uniform(-18/255.0, 18/255.0)
            else:
                adjust_factor = random.uniform(0.5, 1.5)

            new_image = d(new_image, adjust_factor)

    return new_image


def box_expand(image, boxes, filler):
    old_h = image.size(1)
    old_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)

    new_h = int(scale * old_h)
    new_w = int(scale * old_w)


    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(2)

    left = random.randint(0, new_w - old_w)
    right = left+old_w
    top = random.randint(0, new_h - old_h)
    bottom = top+old_h
    new_image[:, top:bottom, left:right] = image

    new_boxes = boxes + torch.FloatTensor([left, top, left, top]) # broadcasting

    return new_image, new_boxes

def box_random_crop(image, boxes, labels, difficults):
    old_h = image.size(1)
    old_w = image.size(2)

    choices_for_overlap = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None]

    while True:
        min_overlap = random.choice(choices_for_overlap)

        if min_overlap is None:
            return image, boxes, labels, difficults


        max_trials = 50

        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1.0)
            scale_w = random.uniform(min_scale, 1.0)

            new_h = int(scale_h * old_h)
            new_w = int(scale_w * old_w)

            aspect_ratio = new_h/new_w

            if aspect_ratio<=0.5 or aspect_ratio>=2.0:
                continue


            left = random.randint(0, old_w-new_w)
            right = left+new_w
            top = random.randint(0, old_h-new_h)
            bottom = top+new_h

            crop = torch.FloatTensor([left, top, right, bottom])

            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes).squeeze(0)

            if overlap.max().item() < min_overlap:
                continue

            # crop image
            new_image = image[:,top:bottom, left:right]

            # find valid boxes, i.e. boxes with centers in the crop region

            bb_centers = boxes[:,:2]+boxes[:,2:]
            bb_centers = bb_centers/2.0

            centers_in_crop = (bb_centers[:,0]>left)*(bb_centers[:,0]<right)*(bb_centers[:,1]>top)*(bb_centers[:,1]<bottom)

            if not centers_in_crop.any():
                continue


            new_boxes = boxes[centers_in_crop,:]
            new_labels = labels[centers_in_crop]
            new_difficults = difficults[centers_in_crop]

            # fix the bboxes's coordinates

            new_boxes[:, :2] = torch.max(new_boxes[:,:2], crop[:2])
            new_boxes[:, :2] -= crop[:2]

            new_boxes[:, 2:] = torch.min(new_boxes[:,2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficults



def box_flip(image, boxes):
    new_image = FT.hflip(image)
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    return new_image, new_boxes[:, [2, 1, 0, 3]]

def transform(image, boxes, labels, difficults, split='TRAIN', resize_dims=(300, 300)):
    assert split in {"TRAIN","TEST"}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficults = difficults


    if split == 'TRAIN':
        new_image = photometric_distort(new_image)
        new_image = FT.to_tensor(new_image)

        if random.random()<0.5:
            new_image, new_boxes = box_expand(new_image, new_boxes, filler=mean)

        new_image, new_boxes, new_labels, new_difficults = box_random_crop(new_image, new_boxes, new_labels, new_difficults)
        new_image = FT.to_pil_image(new_image)
        if random.random()<0.5:
            new_image, new_boxes = box_flip(new_image, new_boxes)

    new_image, new_boxes = box_resize(new_image, new_boxes, dims=resize_dims)

    new_image = FT.to_tensor(new_image)
    new_image = FT.normalize(new_image, mean, std)
    return new_image, new_boxes, new_labels, new_difficults

def test_image_transform(image, resize_dims=(300,300)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image = FT.resize(image, resize_dims)
    new_image = FT.to_tensor(new_image)
    new_image = FT.normalize(new_image, mean, std)
    return new_image