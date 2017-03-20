# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" A kaggle competiton about recognise fishes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'fish_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 11002,
                   'validation': 2750,
                   'test': 1000}

_NUM_CLASSES = 8

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'The ID of the image',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each objec, between 0 and 7',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading fish.

    Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

    Returns:
    A `Dataset` namedtuple.

    Raises:
    ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
        file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/height': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/width': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/class/label': tf.FixedLenFeature([], tf.string, default_value='no name'),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'name': slim.tfexample_decoder.Tensor('image/class/label'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)
