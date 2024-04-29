from __future__ import division
from __future__ import print_function


import numpy as np
import pickle
import random
import sys

import torch
class Dataset(object):
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        self._saveIDs = self.saveIDs()

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        #self._perm = None
        self._perm = np.arange(self._num_examples)
        np.random.shuffle(self._perm)
    def reinitialize_index(self):
        self._index_in_epoch = 0
        return None
    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        np.random.shuffle(self._saveIDs)
        return self._saveIDs

    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # flowers dataset
            class_name = 'class_%05d/' % class_id
            name = name.replace('jpg/', class_name)
        cap_path = '%s/text_c10/%s.txt' %\
                   (self.workdir, name)
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def transform(self, images):
        if self._aug_flag:
            transformed_images =\
                np.zeros([images.shape[0], self._imsize, self._imsize, 3])
            for i in range(images.shape[0]):
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(images[i])
                else:
                    transformed_images[i] = images[i]
            return transformed_images
        else:
            return images

    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num,
                                          sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    captions = self.readCaptions(filenames[i],
                                                 class_id[i])
                    #sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        collision_flag =\
            (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] +
             np.random.randint(100, 200)) % self._num_examples

        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [torch.FloatTensor(sampled_images.transpose((0,3,1,2))), torch.FloatTensor(sampled_wrong_images.transpose((0,3,1,2)))]

        if self._embeddings is not None:
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, class_id, window)
            ret_list.append(torch.FloatTensor(sampled_embeddings))
            ret_list.append(torch.FloatTensor(sampled_captions))
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(torch.LongTensor(np.array(self._labels)[current_ids]-1))
        else:
            ret_list.append(None)
        return ret_list
    def next_batch_test(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            ret_list = []
            return ret_list
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        collision_flag =\
            (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] +
             np.random.randint(100, 200)) % self._num_examples

        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [torch.FloatTensor(sampled_images.transpose((0,3,1,2))), torch.FloatTensor(sampled_wrong_images.transpose((0,3,1,2)))]

        if self._embeddings is not None:
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, class_id, window)
            ret_list.append(torch.FloatTensor(sampled_embeddings))
            ret_list.append(torch.FloatTensor(sampled_captions))
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(torch.LongTensor(np.array(self._labels)[current_ids]-1))
        else:
            ret_list.append(None)
        return ret_list
    def next_batch_val(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            sys.exit()
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        collision_flag =\
            (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] +
             np.random.randint(100, 200)) % self._num_examples

        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [torch.FloatTensor(sampled_images.transpose((0,3,1,2))), torch.FloatTensor(sampled_wrong_images.transpose((0,3,1,2)))]

        if self._embeddings is not None:
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, class_id, window)
            ret_list.append(torch.FloatTensor(sampled_embeddings))
            ret_list.append(torch.FloatTensor(sampled_captions))
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(torch.LongTensor(np.array(self._labels)[current_ids]-1))
        else:
            ret_list.append(None)
        return ret_list


class TextDataset(object):
    def __init__(self, workdir, embedding_type, image_size):
        self.image_filename = '/128images.pickle'


        self.image_shape = [image_size,
                            image_size, 3]
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None
        self.train = None
        self.test = None
        self.workdir = workdir
        if embedding_type == 'cnn-rnn':
            self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            self.embedding_filename = '/skip-thought-embeddings.pickle'

    def get_data(self, pickle_path, aug_flag=True):
        with open(pickle_path + self.image_filename, 'rb') as f:
            images = pickle.load(f, encoding='latin1')
            images = np.array(images)
            print('images: ', images.shape)

        with open(pickle_path + self.embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            self.embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f, encoding='latin1')
            print('list_filenames: ', len(list_filenames), list_filenames[0])
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding='latin1')

        return Dataset(images, self.image_shape[0], embeddings,
                       list_filenames, self.workdir, class_id,
                       aug_flag, class_id)
