#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : reasoning.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import functools
from typing import Optional, Sequence, Dict

import jactorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['LeftGrounding']


class VisualConceptInferenceCache(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = dict()

    def is_cached(self, *args):
        return args in self._cache

    def get_cache(self, *args):
        return self._cache[args]

    def set_cache(self, *args, value=None):
        self._cache[args] = value
        return value

    @staticmethod
    def cached_result(cache_key):
        def wrapper(func):
            @functools.wraps(func)
            def wrapped(self, *args):
                if self.is_cached(cache_key, *args):
                    return self.get_cache(cache_key, *args)

                value = func(self, *args)
                return self.set_cache(cache_key, *args, value=value)
            return wrapped
        return wrapper


class LeftGrounding(VisualConceptInferenceCache):
    def __init__(self, raw_features, embedding_registry, training, attribute_class_to_idx, input_objects_class):
        super().__init__()
        self.raw_features = raw_features
        self.embedding_registry = embedding_registry
        self.train(training)
        self.attribute_class_to_idx = attribute_class_to_idx
        self.input_objects_class = input_objects_class

    def ones_value(self, *shape, log=True):
        offset = 10 if log else 1
        tensor = torch.zeros(shape, dtype=torch.float32, device=self.get_device()) + offset
        return tensor

    def get_device(self):
        return self.raw_features['attribute'].device

    def get_nr_objects(self):
        return self.raw_features['attribute'].size(0)

    @VisualConceptInferenceCache.cached_result('compute_similarity')
    def compute_similarity(self, concept_cat, concept):
        if concept_cat == 'attribute':
            idx = self.attribute_class_to_idx[concept]
            mask = self.raw_features[concept_cat][:, idx]

        elif concept_cat == 'relation':
            mask = self.get_embedding_mod(concept_cat).similarity(self.raw_features[concept_cat], concept)
        else:
            feat = self.raw_features[concept_cat]
            object_len = feat.size(0)
            feat = torch.cat([
                jactorch.add_dim(feat, 0, object_len),
                jactorch.add_dim(feat, 1, object_len),
                jactorch.add_dim(feat, 2, object_len)
            ], dim=3)
            mask = self.get_embedding_mod(concept_cat).similarity(feat, concept)
        return mask

    def compute_description(self, concept_cat: str, attribute: str):
        # TODO(Joy Hsu @ 2023/03/27): note that this only works for the HumanMotionQA dataset.
        if concept_cat == 'attribute':
            from concepts.benchmark.vision_language.babel_qa.humanmotion_constants import attribute_concepts_mapping
            this_attribute_concepts = attribute_concepts_mapping[attribute]

            output_vocab = self.embedding_registry.output_vocab.idx2word
            attribute_features = self.raw_features['attribute']
            all_scores = []
            for idx in range(len(output_vocab)):
                word = output_vocab[idx].replace(' ', '_') + '_Action'
                if word in this_attribute_concepts:
                    scores = self.compute_similarity(concept_cat, word)  # [num_segments]
                else:
                    scores = torch.zeros(attribute_features.size(0), device=attribute_features.device) - 100
                all_scores.append(scores)
            all_scores = torch.stack(all_scores, dim=1)  # [num_segments, output_vocab_len]

            return all_scores
        # e.g., what's the color of the object? compute_description('attribute', 'color')
        embedding_mod = self.get_embedding_mod(concept_cat)
        return embedding_mod.compute_similarity(self.raw_features[concept_cat], attribute)

    def get_embedding_mod(self, concept_cat):
        embedding_mod_name = concept_cat + '_embedding'
        return getattr(self.embedding_registry, embedding_mod_name)


class NCDenseClipGrounding(VisualConceptInferenceCache):
    def __init__(self, raw_features, embedding_registry, training, attribute_class_to_idx, pick_attn, goal_transport, denseclip):
        super().__init__()
        self.train(training)

        self.raw_features = raw_features
        self.embedding_registry = embedding_registry
        self.training = training
        self.attribute_class_to_idx = attribute_class_to_idx

        self.pick_attn = pick_attn
        self.goal_transport = goal_transport
        self.denseclip = denseclip

    @VisualConceptInferenceCache.cached_result('compute_similarity')
    def compute_similarity(self, concept_cat, concept):
        img = self.raw_features['img']
        lang = concept.split('_Object')[0].replace('_', ' ')  # Check what natural language form is passed in (e.g. pepsi next box)
        im_attn_map = self.denseclip(img, lang)
        im_attn_map = im_attn_map.permute(1, 2, 0)
        im_attn_map = im_attn_map.flatten()  # Flatten for softmax computation
        return im_attn_map

    def compute_action(self, pick_mask, place_mask, action_concept):
        action_concept = action_concept.split('_Action_Object_Object')[0].replace('_', ' ')  # e.g. pack in
        pick_mask = torch.reshape(pick_mask.tensor, (320, 160, -1))  # Reshape to 2D mask
        place_mask = torch.reshape(place_mask.tensor, (320, 160, -1))

        if self.training:
            # Pick
            img = self.raw_features['img']
            pick_attention_scores = self.pick_attn.forward(img, pick_mask.detach(), action_concept, softmax=False)  # Check softmax
            pick_attention_scores = pick_attention_scores.reshape(1, pick_mask.shape[0], pick_mask.shape[1]).permute(1, 2, 0)

            # Place
            pick_center = self.raw_features['p0']
            place_out = self.goal_transport.ns_forward(img, place_mask.detach(), pick_center, action_concept, pad_w=0, pad_h=0, softmax=False)
            return pick_attention_scores, place_out  # [320, 160, 1], [1, 36, 320, 160]
        else:
            action = {}
            # Pick
            img = self.raw_features['img']
            pick_attention_scores = self.pick_attn.forward(img, pick_mask.detach(), action_concept, softmax=False).detach()
            pick_attention_scores = pick_attention_scores.reshape(1, pick_mask.shape[0], pick_mask.shape[1]).permute(1, 2, 0)
            pick_map = pick_attention_scores.detach().cpu().numpy()
            argmax = np.argmax(pick_map)
            argmax = np.unravel_index(argmax, shape=pick_map.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_map.shape[2])

            action['p0_pix'] = p0_pix
            action['p0_theta'] = p0_theta

            # Place
            pick_center = p0_pix
            place_out = self.goal_transport.ns_forward(img, place_mask.detach(), pick_center,
                                                       action_concept, pad_w=0, pad_h=0, softmax=False).detach()
            place_out = place_out.squeeze(0)  # Batch_size 1 assumption
            place_out = place_out.permute(1, 2, 0)
            place_out = place_out.detach().cpu().numpy()
            argmax = np.argmax(place_out)
            argmax = np.unravel_index(argmax, shape=place_out.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_out.shape[2])

            action['p1_pix'] = p1_pix
            action['p1_theta'] = p1_theta
            return action, pick_map, place_out


class NCOneTimeComputingGrounding(VisualConceptInferenceCache):
    def __init__(self, raw_features, embedding_registry, training: bool, apply_relation_mask: bool, attribute_concepts: Optional[Dict[str, Sequence[str]]] = None, learned_belong_fusion: str = 'min'):
        super().__init__()
        self.raw_features = raw_features
        self.embedding_registry = embedding_registry
        self.train(training)
        self.apply_relation_mask = apply_relation_mask
        self.attribute_concepts = attribute_concepts
        self.learned_belong_fusion = learned_belong_fusion

        assert self.learned_belong_fusion in ('min', 'plus')

        if hasattr(self.embedding_registry, 'concept2output_lhs'):
            self.concept2output_lhs = self.embedding_registry.concept2output_lhs
            self.concept2output_rhs = self.embedding_registry.concept2output_rhs
        else:
            attribute_embedding = self.embedding_registry.attribute_embedding
            if attribute_embedding.tied_attributes:
                output_vocab = self.embedding_registry.output_vocab

                concept2output_lhs = list()
                concept2output_rhs = list()
                for word in output_vocab.word2idx:
                    concept = word + '_Object'
                    if concept in attribute_embedding.concept2id:
                        concept2output_lhs.append(attribute_embedding.concept2id[concept])
                        concept2output_rhs.append(output_vocab.word2idx[word])

                self.concept2output_lhs = torch.tensor(concept2output_lhs, dtype=torch.long, device=attribute_embedding.attribute_belongings.device)
                self.concept2output_rhs = torch.tensor(concept2output_rhs, dtype=torch.long, device=attribute_embedding.attribute_belongings.device)

                # Write back to embedding registry
                self.embedding_registry.concept2output_lhs = self.concept2output_lhs
                self.embedding_registry.concept2output_rhs = self.concept2output_rhs

    def ones_value(self, *shape, log=True):
        offset = 10 if log else 1
        tensor = torch.zeros(shape, dtype=torch.float32, device=self.get_device()) + offset
        return tensor

    def get_device(self):
        return self.raw_features['attribute'].device

    def get_nr_objects(self):
        return self.raw_features['attribute'].size(0)

    @VisualConceptInferenceCache.cached_result('compute_all_similarity')
    def compute_all_similarity(self, concept_cat: str):
        # if concept_cat == 'attribute':
        #     return self._compute_groundtruth_similarity_attribute()
        # if concept_cat == 'relation':
        #     return self._compute_groundtruth_similarity_relation()
        # assert False

        similarity = self.get_embedding_mod(concept_cat).compute_all_similarity(self.raw_features[concept_cat])
        if self.apply_relation_mask and concept_cat == 'relation':
            similarity = _do_apply_self_mask(similarity)
        return similarity

    def compute_similarity(self, concept_cat: str, concept: str):
        embedding_mod = self.get_embedding_mod(concept_cat)
        return self.compute_all_similarity(concept_cat)[..., embedding_mod.concept2id[concept]]

    def compute_similarities_batch(self, concept_cat: str, concept: Sequence[str]):
        embedding_mod = self.get_embedding_mod(concept_cat)
        indices = [embedding_mod.concept2id[c] for c in concept]
        return self.compute_all_similarity(concept_cat)[..., indices]

    def compute_description(self, concept_cat: str, attribute: str):
        # TODO(Jiayuan Mao @ 2023/03/23): temporally putting things here. But note that this only works for the CLEVR dataset.

        if concept_cat == 'attribute':
            if not self.embedding_registry.attribute_embedding.tied_attributes:
                if self.attribute_concepts is not None:
                    if attribute not in self.attribute_concepts:
                        raise ValueError('attribute {} is not in attribute_concepts'.format(attribute))
                    raw_scores = self.compute_similarities_batch(concept_cat, [f'{x}_Object' for x in self.attribute_concepts[attribute]])
                    output_vocab = self.embedding_registry.output_vocab

                    scores = torch.zeros(raw_scores.size(0), len(output_vocab), device=raw_scores.device) - 100
                    for i, concept in enumerate(self.attribute_concepts[attribute]):
                        scores[:, output_vocab.word2idx[concept]] = raw_scores[:, i]
                    return scores
            else:
                similarities = self.compute_all_similarity(concept_cat)
                if self.learned_belong_fusion == 'min':
                    fusion_func = torch.min
                elif self.learned_belong_fusion == 'plus':
                    fusion_func = lambda x, y: x + y
                else:
                    raise ValueError('Unknown learned_belong_fusion {}'.format(self.learned_belong_fusion))

                similarities = fusion_func(
                    similarities,
                    self.embedding_registry.attribute_embedding.attribute_belongings[
                        self.embedding_registry.attribute_embedding.attribute2id[attribute]
                    ][None]
                )
                output_vocab = self.embedding_registry.output_vocab
                scores = torch.zeros(similarities.size(0), len(output_vocab), device=similarities.device) - 100
                scores[:, self.concept2output_rhs] = similarities[:, self.concept2output_lhs]
                return scores

        embedding_mod = self.get_embedding_mod(concept_cat)
        return embedding_mod.compute_description(self.raw_features[concept_cat], attribute)

    def get_embedding_mod(self, concept_cat):
        embedding_mod_name = concept_cat + '_embedding'
        return getattr(self.embedding_registry, embedding_mod_name)

    # def _compute_groundtruth_similarity_attribute(self):
    #     scene = self.scene
    #     concept2id = self.get_embedding_mod('attribute').concept2id
    #     vals = torch.zeros((self.get_nr_objects(), len(concept2id)), dtype=torch.float32, device=self.get_device())
    #     vals[:, concept2id['thing_Object']] = 1
    #     for attr_name, concepts in self.attribute_concepts.items():
    #         for i, obj in enumerate(scene['objects']):
    #             assert attr_name in obj
    #             vals[i, concept2id[obj[attr_name] + '_Object']] = 1
    #     vals = -100 + 200 * vals
    #     return vals

    # def _compute_groundtruth_similarity_relation(self):
    #     scene = self.scene
    #     concept2id = self.get_embedding_mod('relation').concept2id
    #     vals = torch.zeros((self.get_nr_objects(), self.get_nr_objects(), len(concept2id)), dtype=torch.float32, device=self.get_device())

    #     for attr_name in self.attribute_concepts:
    #         for i, obj1 in enumerate(scene['objects']):
    #             for j, obj2 in enumerate(scene['objects']):
    #                 if i == j:
    #                     continue
    #                 vals[i, j, concept2id[f'same_{attr_name}_Object_Object']] = obj1[attr_name] == obj2[attr_name]
    #                 if f'equal_{attr_name}_Object_Object' in concept2id:
    #                     vals[i, j, concept2id[f'equal_{attr_name}_Object_Object']] = obj1[attr_name] == obj2[attr_name]
    #     for _, relations in self.relational_concepts:
    #         for relation in relations:
    #             for i, obj1 in enumerate(scene['objects']):
    #                 for j, obj2 in enumerate(scene['objects']):
    #                     if i == j:
    #                         continue
    #                     vals[i, j, concept2id[relation]] = (i in scene['relationships'][relation][j])
    #     vals = -100 + 200 * vals
    #     return vals


def _get_self_mask(m):
    self_mask = torch.eye(m.size(0), dtype=m.dtype, device=m.device)
    return self_mask


def _do_apply_self_mask(m):
    self_mask = _get_self_mask(m)
    if len(m.shape) == 2:
        return m * (1 - self_mask) + (-10) * self_mask
    else:
        return m * (1 - self_mask.unsqueeze(-1)) + (-10) * self_mask.unsqueeze(-1)


def _logit_softmax(tensor, dim=-1):
    return F.log_softmax(tensor, dim=dim) - torch.log1p(-F.softmax(tensor, dim=dim).clamp(max=1 - 1e-8))

