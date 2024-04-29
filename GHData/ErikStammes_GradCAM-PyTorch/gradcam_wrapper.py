from inspect import isclass

import torch
import torch.nn.functional as F

def gradcam_wrapper(model):
    """ Wrapper that can generate Grad-CAMs, which dynamically inherits from the given argument model

    Args:
        model: A non-initialized PyTorch model of type torch.nn.Module.

    Return:
        model: A wrapped version of your model. It now accepts an extra (optional) argument `labels` in the forward()
                function, which is used to generate the Grad-CAMs. The forward() function now also returns a tuple of
                (logits, gcams)

    """
    if not isclass(model):
        if isinstance(model, torch.nn.Module):
            message = 'Given model is not a class but an instance of torch.nn.Module. ' \
                       + 'Do not instantiate your model before wrapping it using this function.'
        else:
            message = 'Given model is not a class.'
        raise TypeError(message)

    class GradCAMWrapper(model):
        """ Wrapper that stores the forward features and gradients to create a GradCAM heatmap / mask """
        def __init__(self, model_params, gradient_layer, multitarget=False, multitarget_threshold=0.5, img_shape=None):
            """
            constructor for the  GradCAM-wrapped model
            :param model_params: the params to initialize the original model
            :param gradient_layer: the gradient layer to compute the Grad-CAM at
            :param multitarget: (optional) if your model uses multiple targets per image  #TODO: add threshold
            :param multitarget_threshold (optional) threshold to use on the logits to select the predicted classes
            :param img_shape: (optional) output shape [H, W] of the GradCAM.
            """
            super(GradCAMWrapper, self).__init__(**model_params)
            self.gradient_layer = gradient_layer
            self.multitarget = multitarget
            self.multitarget_threshold = multitarget_threshold
            self.img_shape = img_shape

            # Initialize properties to store features during forward/backward pass in
            self.forward_features = None
            self.backward_features = None

            self._register_hooks()

        def _register_hooks(self):
            """ Register forward and backward hooks that store features and gradients from the given layer """
            def forward_hook(_module, _forward_input, forward_output):
                self.forward_features = forward_output

            def backward_hook(_module, _backward_input, backward_output):
                self.backward_features = backward_output[0]

            for name, module in self.named_modules():
                if name == self.gradient_layer:
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)
                    break

        def _one_hot_encode(self, tensor, shape):
            """ One hot encode a tensor """
            tensor = tensor.view(-1, 1)
            one_hot = torch.zeros(shape, device=tensor.device)
            one_hot.scatter_(1, tensor, 1)
            return one_hot

        def _compute_gradcam(self, device):
            """ Computes the GradCAM heatmaps """
            assert self.backward_features, 'GradCAM can only be computed after the gradients have been computed'
            weights = F.adaptive_avg_pool2d(self.backward_features[device], 1)
            gcam = torch.mul(self.forward_features[device], weights).sum(dim=1, keepdim=True)
            gcam = F.relu(gcam)
            gcam = F.interpolate(gcam, self.img_shape, mode='bilinear', align_corners=False)
            batch_size, channel_size, height, width = gcam.shape
            gcam = gcam.view(batch_size, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam_max = gcam.max(dim=1, keepdim=True)[0]
            gcam /= torch.where(gcam_max != 0, gcam_max, torch.ones_like(gcam_max))
            gcam = gcam.view(batch_size, channel_size, height, width)
            return gcam

        def forward(self, images, labels=None):
            """ Extracts the GradCAM after forwarding the input images """
            if self.multitarget and images.size(0) != 1:
                raise Exception('Multitarget Grad-CAM only works with a batch size of 1')

            # Do a forward pass using the original models forward function
            logits = super().forward(images)
            if self.img_shape is None:
                self.img_shape = images.size()[2:]

            if self.multitarget:
                if labels is None:
                    # Create a tensor of labels
                    labels = (logits > self.multitarget_threshold).nonzero()
            else:
                if labels is None:
                    labels = logits.argmax(dim=1)
                labels = [labels]

            gcams = []
            # Iterate over the labels, if not using multitarget this loop runs just once
            for label in labels:
                gradient = self._one_hot_encode(label, logits.shape)
                logits.backward(gradient=gradient, retain_graph=True)

                gcam = self._compute_gradcam(images.device)
                gcams.append(gcam)

            if len(gcams) == 1:
                return logits, gcams[0]
            else:
                return logits, torch.as_tensor(gcams)
    return GradCAMWrapper
