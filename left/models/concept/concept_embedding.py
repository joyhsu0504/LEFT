#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : concept_embedding.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle
import jactorch

__all__ = ['NCVSEConceptEmbedding', 'NCLinearConceptEmbedding']


class _AttributeCrossBlock(nn.Module):
    def __init__(self, name, embedding_dim):
        super().__init__()

        self.name = name
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(embedding_dim * 4, 1)

    def forward(self, a, b=None):
        if b is None:
            a, b = jactorch.meshgrid(a, dim=-2)

        c = torch.cat((a, b, a * b, a - b), dim=-1)
        return self.embedding(c).squeeze(-1)


class _ConceptBlock(nn.Module):
    """Concept as an embedding in the corresponding attribute space."""

    def __init__(self, name, embedding_dim, nr_attributes):
        """

        Args:
            name (str): name of the concept.
            embedding_dim (int): dimension of the embedding.
            nr_attributes (int): number of known attributes.
        """
        super().__init__()

        self.name = name
        self.embedding_dim = embedding_dim
        self.nr_attributes = nr_attributes
        self.embedding = nn.Parameter(torch.randn(embedding_dim))

        self.belong = nn.Parameter(torch.randn(nr_attributes) * 0.1)
        self.known_belong = False

    def set_belong(self, belong_id):
        """
        Set the attribute that this concept belongs to.

        Args:
            belong_id (int): the id of the attribute.
        """
        self.belong.data.fill_(-100)
        self.belong.data[belong_id] = 100
        self.belong.requires_grad = False
        self.known_belong = True

    @property
    def log_normalized_belong(self):
        """Log-softmax-normalized belong vector."""
        return F.log_softmax(self.belong, dim=-1)

    @property
    def normalized_belong(self):
        """Softmax-normalized belong vector."""
        return F.softmax(self.belong, dim=-1)


class VisualConceptEmbeddingBase(nn.Module):
    def similarity(self, query, identifier):
        raise NotImplementedError()

    def query_attribute(self, query, identifier):
        raise NotImplementedError()


class NCVSEConceptEmbedding(VisualConceptEmbeddingBase):
    """VSE-style concept embeddings. This is used in the original NS-CL paper, and is based on the visual-semantic embeddings."""

    def __init__(self, enable_cross_similariy=False):
        super().__init__()

        self.all_attributes = list()
        self.all_concepts = list()
        self.concept_embeddings = nn.Module()

        self.enable_cross_similariy = enable_cross_similariy
        self.attribute_cross_embeddings = nn.Module()

    @property
    def nr_attributes(self):
        return len(self.all_attributes)

    @property
    def nr_concepts(self):
        return len(self.all_concepts)

    @jacinle.cached_property
    def attribute2id(self):
        return {a: i for i, a in enumerate(self.all_attributes)}

    def init_attribute(self, identifier, input_dim=None):
        assert self.nr_concepts == 0, 'Can not register attributes after having registered any concepts.'
        self.all_attributes.append(identifier)
        self.all_attributes.sort()

        if self.enable_cross_similariy:
            assert input_dim is not None
            block = _AttributeCrossBlock(identifier, input_dim)
            self.attribute_cross_embeddings.add_module('attribute_' + identifier, block)

    def init_concept(self, identifier, input_dim, known_belong=None):
        block = _ConceptBlock(identifier, input_dim, self.nr_attributes)
        self.concept_embeddings.add_module('concept_' + identifier, block)
        self.all_concepts.append(identifier)
        if known_belong is not None:
            block.set_belong(self.attribute2id[known_belong])

    def get_belongs(self):
        """Return a dict which maps from all attributes (by name) to a list of concepts (by name)."""
        belongs = dict()
        for k, v in self.concept_embeddings.named_children():
            belongs[k] = self.all_attributes[v.belong.argmax(-1).item()]
        class_based = dict()
        for k, v in belongs.items():
            class_based.setdefault(v, list()).append(k)
        class_based = {k: sorted(v) for k, v in class_based.items()}
        return class_based

    def get_attribute_cross(self, identifier):
        return getattr(self.attribute_cross_embeddings, 'attribute_' + identifier)

    def get_concept(self, identifier):
        return getattr(self.concept_embeddings, 'concept_' + identifier)

    def get_all_concepts(self):
        return {c: self.get_concept(c) for c in self.all_concepts}

    def get_concepts_by_attribute(self, identifier):
        return self.get_all_concepts(), self.attribute2id[identifier]

    def similarity(self, query, identifier):
        concept = self.get_concept(identifier)
        reference = jactorch.add_dim_as_except(concept.embedding, query, -1)
        logits = (query * reference).sum(dim=-1)
        return logits

    def cross_similarity(self, query, identifier):
        mapping = self.get_attribute_cross(identifier)
        logits = mapping(query)
        return logits

    def query_attribute(self, query, identifier):
        # TODO(Jiayuan Mao @ 01/19): accelerate this.
        concepts, attr_id = self.get_concepts_by_attribute(identifier)

        word2idx = {}
        masks = []
        for k, v in concepts.items():
            embedding = jactorch.add_dim_as_except(v.embedding, query, -1)
            mask = (query * embedding).sum(dim=-1)

            belong_score = v.log_normalized_belong[attr_id]
            mask = mask + belong_score

            masks.append(mask)
            word2idx[k] = len(word2idx)

        masks = torch.stack(masks, dim=-1)
        return masks, word2idx


class NCLinearConceptEmbedding(nn.Module):
    def __init__(self, tied_attributes: bool = False):
        """Initialize the concept embedding.

        Args:
            tied_attributes (bool): whether to tie the attribute query to concept similarities. If True, for each attribute,
                there will be a single vector a, and a[i] = the probability that the concept belongs to the attribute.
        """
        super().__init__()
        self.tied_attributes = tied_attributes
        self.attributes = list()
        self.concepts = list()

        self.linear_input_dim = None
        self.linear_output_dim = 0  # Equal to the number of concepts.
        self.attribute_output_dim = None
        self.register_module('linear', None)

    def extra_state_dict(self):
        return {'attributes': self.attributes.copy(), 'concepts': self.concepts.copy(), 'linear_input_dim': self.linear_input_dim, 'attribute_output_dim': self.attribute_output_dim}

    def load_extra_state_dict_and_weights(self, state_dict, weights):
        if self.linear_input_dim != state_dict['linear_input_dim'] and self.linear_input_dim is not None and state_dict['linear_input_dim'] is not None:
            raise RuntimeError('The linear input dimension is not consistent: {} vs {}.'.format(self.linear_input_dim, state_dict['linear_input_dim']))
        if self.attribute_output_dim != state_dict['attribute_output_dim'] and self.attribute_output_dim is not None and state_dict['attribute_output_dim'] is not None:
            raise RuntimeError('The attribute output dimension is not consistent: {} vs {}.'.format(self.attribute_output_dim, state_dict['attribute_output_dim']))

        from jacinle.config.g import g

        for i, attribute_name in enumerate(state_dict['attributes']):
            if attribute_name not in self.attributes:
                raise RuntimeError(f'The attributes are not consistent: {attribute_name} is not found.')
            if self.tied_attributes:
                self.attribute_belongings.data[self.attribute2id[attribute_name], :] = -100
                for j, concept in enumerate(state_dict['concepts']):
                    if concept in self.concepts:
                        self.attribute_belongings.data[self.attribute2id[attribute_name], self.concept2id[concept]] = weights['attribute_belongings'][i, j]
            else:
                # Should have been handled by the outer loop.
                # NB(Jiayuan Mao @ 2023/04/30): the directly mapped concepts should have been handled by the outer loop; For the reset, they should still be learned.
                pass

        for i, concept_name in enumerate(state_dict['concepts']):
            if concept_name not in self.concepts:
                raise RuntimeError(f'The concepts are not consistent: {concept_name} is not found.')

            self.linear.weight.data[self.concept2id[concept_name], :] = weights['linear.weight'][i, :]
            self.linear.bias.data[self.concept2id[concept_name]] = weights['linear.bias'][i]

        if g.concept_mapping is not None:
            for c1, c2 in g.concept_mapping.items():
                if f'{c1}_Object' in self.concepts and f'{c2}_Object' in self.concepts:
                    print('Mapping {} <- {}'.format(c1, c2))
                    self.linear.weight.data[self.concept2id[f'{c1}_Object'], :] = self.linear.weight.data[self.concept2id[f'{c2}_Object'], :]
                    self.linear.bias.data[self.concept2id[f'{c1}_Object']] = self.linear.bias.data[self.concept2id[f'{c2}_Object']]

    @jacinle.cached_property
    def concept2id(self):
        return {c: i for i, c in enumerate(self.concepts)}

    @jacinle.cached_property
    def attribute2id(self):
        return {a: i for i, a in enumerate(self.attributes)}

    def init_concept(self, identifier, input_dim):
        self._set_linear_input_dim(input_dim)
        self.concepts.append(identifier)
        self.linear_output_dim += 1

    def init_attribute(self, identifier, input_dim, output_dim):
        self._set_linear_input_dim(input_dim)
        self._set_attribute_output_dim(output_dim)
        self.attributes.append(identifier)

    def _set_linear_input_dim(self, input_dim: int):
        if self.linear_input_dim is None:
            self.linear_input_dim = input_dim
        else:
            assert self.linear_input_dim == input_dim, 'All concepts must have the same input dimension.'

    def _set_attribute_output_dim(self, output_dim: int):
        if self.attribute_output_dim is None:
            self.attribute_output_dim = output_dim
        else:
            assert self.attribute_output_dim == output_dim, 'All attributes must have the same output dimension (aka the size of the vocabulary).'

    def init_linear_layers(self):
        assert self.linear is None, 'Linear layer has already been initialized.'
        if self.linear_input_dim is not None and self.linear_output_dim > 0:
            self.linear = nn.Linear(self.linear_input_dim, self.linear_output_dim)
        if self.tied_attributes:
            self.attribute_belongings = nn.Parameter(torch.zeros(len(self.attributes), len(self.concepts)))
        else:
            for attribute in self.attributes:
                self.register_module('attribute_' + attribute, nn.Linear(self.linear_input_dim, self.attribute_output_dim))

    def compute_all_similarity(self, query):
        assert self.linear is not None, 'Must call init_linear_layers() first.'
        logits = self.linear(query)
        return logits

    def compute_description(self, query, identifier):
        assert self.linear is not None, 'Must call init_linear_layers() first.'
        logits = getattr(self, 'attribute_' + identifier)(query)
        return logits

