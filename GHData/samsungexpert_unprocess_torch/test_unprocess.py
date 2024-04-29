import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

def random_ccm():
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                 [-0.5625, 1.6328, -0.0469],
                 [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                 [-0.613, 1.3513, 0.2906],
                 [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                 [-0.2887, 1.0725, 0.2496],
                 [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                 [-0.4782, 1.3016, 0.1933],
                 [-0.097, 0.1581, 0.5181]]]

    num_ccm = len(xyz2cams)
    xyz2cams = torch.Tensor(xyz2cams)
    weights = torch.rand(num_ccm,1,1)
    weight_sum = torch.sum(weights, axis=0)

    xyz2cam = torch.sum(xyz2cams * weights, axis=0) / weight_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.Tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = xyz2cam * rgb2xyz

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, axis=-1, keepdims=True)
    return rgb2cam

def random_gains():
  # RGB gain represents brightening.
  rgb_gain = 1.0/ torch.normal(0.8,0.1,(1,1)).squeeze().unsqueeze(dim=-1)

  # Red and blue gains represent white balance.
  #red_gain = tf.random_uniform((), 1.9, 2.4)
  red_gain = 1.9 + 0.5*torch.rand(1)
  #blue_gain = tf.random_uniform((), 1.5, 1.9)
  blue_gain = 1.5 + 0.4*torch.rand(1)

  return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image):
    ## Approximately inverts a global tone mapping curve."""
    image = torch.clamp(image, min=0.0, max=1.0)

    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)

def gamma_expansion(image):
  """Converts from gamma to linear space."""
  # Clamps to prevent numerical instability of gradients near zero.
  return torch.clamp(image, min=1e-8) ** 2.2


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  shape = image.shape
  image = torch.reshape(image, [3, -1])
  # print(image.shape, ccm.shape)
  image = ccm @ image
  return torch.clamp(torch.reshape(image, shape), min=0., max=1.)


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""

  #print(1./rgb_gain, 1./red_gain, 1./blue_gain)
  gains = torch.stack((1.0 / red_gain, torch.Tensor([1.0]), 1.0 / blue_gain),dim=-1) / rgb_gain

  gains = torch.diag(gains.squeeze())
  # print(gains)

  # Prevents dimming of saturated pixels by smoothly masking gains near white.
  gray = torch.mean(image, axis=0)
  gray = torch.stack([gray, gray,gray], dim=0)

  C, H, W = gray.shape
  inflection = 0.9
  mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
  mask_ones = torch.ones_like(mask)

  mask = mask.view(3,-1)
  mask_ones = mask_ones.view(3,-1)

  image_vec = image.view(3,-1)

  gain_map = torch.maximum(mask+gains@(1.0-mask), gains@mask_ones)
  # gain_map = gains@mask_ones


  image_result = image_vec * gain_map

  return image_result.view(-1,H,W)

def unprocess(image):
    rgb2cam = random_ccm()
    cam2rgb = torch.linalg.pinv(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    #save_temp('inverse_smooth_step.png',image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    #save_temp('inv_gamma.png', image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    #save_temp('ccm.png', image)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    #save_temp('inverse_gains.png', image)
    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)
    # Applies a Bayer mosaic.
    # image = mosaic(image)
    return image

def save_temp(filename,img):
    img_pil = ToPILImage()(img)
    img_pil.save(filename)



if __name__ == "__main__":

    filepath = "img/4.png"

    img = Image.open(filepath).convert('RGB')
    img_tensor = ToTensor()(img)

    unprocess(img_tensor)


