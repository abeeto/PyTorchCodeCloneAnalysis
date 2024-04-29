from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.segmentation import mark_boundaries

def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    with open(address, 'wb+') as f:
      Image.fromarray(image).save(f, format='PNG')

def save_ace_report(scores:dict, address:str):
  """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
  report = '\n\n\t\t\t ---CAV accuracies---'
  for bn in scores.keys():
    report += '\n'
    for concept in scores[bn]['concepts']:
      report += '\n' + bn + ':' + concept + ':' + str(round(scores[bn][concept]['acc'],3))
  with open(address, 'w') as f:
    f.write(report)
  report = '\n\n\t\t\t ---TCAV scores---'
  for bn in scores.keys():
    report += '\n'
    for concept in scores[bn]['concepts']:

      report += '\n{}:{}:{}'.format(bn, concept,scores[bn][concept]['tcav'].item())
  with open(address, 'a') as f:  # 追加写入
    f.write(report)


def plot_concepts(concept_dict,scores, layers, num=10, address=None, mode='diverse', concepts=None, average_image_value=117):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  for bn in layers:
    if concepts is None:
      concepts =  scores[bn]['concepts']
    elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
      concepts = [concepts] # 把concept变成list形式
    num_concepts = len(concepts)  # 计算概念的总个数
    plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
    fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
    outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
    for n, concept in enumerate(concepts):
      inner = gridspec.GridSpecFromSubplotSpec(
          2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
      concept_images = concept_dict[bn][concept]['images']
      concept_patches = concept_dict[bn][concept]['patches']
      concept_image_numbers = concept_dict[bn][concept]['image_numbers']
      if mode == 'max':
        idxs = np.arange(len(concept_images))
      elif mode == 'random':
        idxs = np.random.permutation(np.arange(len(concept_images)))
      elif mode == 'diverse':
        idxs = []
        while True:
          seen = set()
          for idx in range(len(concept_images)):
            if concept_image_numbers[idx] not in seen and idx not in idxs:
              seen.add(concept_image_numbers[idx])
              idxs.append(idx)
          if len(idxs) == len(concept_images):
            break
      else:
        raise ValueError('Invalid mode!')
      idxs = idxs[:num]
      for i, idx in enumerate(idxs):
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(concept_images[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        if i == int(num / 2):
          ax.set_title(concept)
        ax.grid(False)
        fig.add_subplot(ax)
        ax = plt.Subplot(fig, inner[i + num])
        mask = 1 - (np.mean(concept_patches[idx] == float(
            average_image_value) / 255, -1) == 1)
        image = concept_dict['discovery_images'][concept_image_numbers[idx]]
        ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(concept_image_numbers[idx]))
        ax.grid(False)
        fig.add_subplot(ax)
    plt.suptitle(bn)
    if address is not None:
      with open(address + bn + '.png', 'wb') as f:
        fig.savefig(f)
      plt.clf()
      plt.close(fig)