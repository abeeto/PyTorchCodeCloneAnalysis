# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn

"""
DEREK'S COMMENT
Seems like generalized RCNN is basically:
    - a transform performed on the inputs prior to feature extraction, ideally matching
        the transformed performed on the backbone in initial training
    - a backbone feature extractor, which is generally a classification CNN such as vgg or Resnet
    - a region proposal network (see rpn.py for more on this)
    - a number of heads for classification, regression, etc. on each region
    
Some models that inherit from GeneralizedRCNN are:
    - faster_rcnn - for classification and 2D bounding box regression on each region
    - Mask_RCNN - for semantic segmentation of pixels in each region
    - Keypoint_RCNN - for regression of facial keypoints in each region\
    
I'll indicate the rest of my comments with ##
"""
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        ## define the 4 main nn.Module objects that comprise the network
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        ## check to make sure targets are provided if in training mode
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        ## get height and width for each image in input list
        original_image_sizes = [img.shape[-2:] for img in images]
        ## 1. perform transform on each image (passed as a list, conceivably pytorch transforms accomodate this structure in parallel processing)
        images, targets = self.transform(images, targets)
        
        ## 2. get features from CNN backbone
        features = self.backbone(images.tensors)
        ## not exactly sure but I think the idea is to encapsulate features in 
        ## the same data type as targets are presented before passing to rpn
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        ## 3. get region proposals - the output of this is anchors, not cropped regions
        ## an anchor is essentially an x,y, scale and aspect ratio (I believe)
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        ## 4. get the relevant predictions - in the case of FasterRCNN, roi_align
        ## is applied during this step to transform from anchor boxes to crops 
        ## of the features of a consistent size
        ## then, classification or regression, etc. is performed
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        ## conceivably, whatever transform is passed to the constructor should
        ## have a postprocess function that scales outputs relative to original image sizes
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        ## add the keys from detector_losses and proposal_losses to losses
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        ## there are no losses if not in training stage
        ## if in training stage, detections aren't relevant so don't return them
        ## though conceivably if you wanted to do something special like plot detections
        ## periodically you actually would want to return them which this doesn't allow for
        if self.training:
            return losses

        return detections
