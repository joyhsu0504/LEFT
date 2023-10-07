#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : listening_dataset.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/22/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from torch.utils.data import Dataset
from functools import partial
import warnings
import multiprocessing as mp
from torch.utils.data import DataLoader

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from left.data.referit3d.referit3d_reader import decode_stimulus_string


'''From ReferIt3D (https://github.com/referit3d/referit3d)'''


def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n


def dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    batch_size_multiplier = 1 if split == 'train' else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == 'train' and len(dataset) % b_size == 1:
        print('dropping last batch during training')
        drop_last = True

    shuffle = split == 'train'

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    data_loader = DataLoader(dataset,
                             batch_size=b_size,
                             num_workers=n_workers,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn)
    return data_loader


def sample_scan_object(object, n_points):
    sample = object.sample(n_samples=n_points)
    return np.concatenate([sample['xyz'], sample['color']], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.ones(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        instance_labels.extend(['pad'] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in instance_labels])

    return instance_labels


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz
    return segmented_objects


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, include_anchors=True):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation

        # Anchor
        self.include_anchors = include_anchors

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'

        # Anchor
        if self.include_anchors:
            anchor_ids = ref['anchor_ids']
            anchor_ids = anchor_ids.strip('][').split(', ')
            anchor_ids = [int(l) for l in anchor_ids]
            anchors = [scan.three_d_objects[i] for i in anchor_ids]
        else:
            anchors = None

        return scan, target, tokens, is_nr3d, anchors

    def prepare_distractors(self, scan, target, anchors):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        already_included = [target_label]
        if self.include_anchors:
            for anchor in anchors:
                anchor_label = anchor.instance_label
                already_included.append(anchor_label)
                distractors.append(anchor)

        # Then all more objects up to max-number of distractors
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d, anchors = self.get_reference_data(index)

        # Make a context of distractors
        context = self.prepare_distractors(scan, target, anchors)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)

        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['scan_id'] = self.references.loc[index]['scan_id']
        res['utterance'] = self.references.loc[index]['utterance']
        res['stimulus_id'] = self.references.loc[index]['stimulus_id']


        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate')

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
