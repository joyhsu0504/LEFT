#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_left.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import functools

import jacinle.io as io
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.config.environ_v2 import configs, set_configs
from left.models.model import LeftModel
from left.domain import make_domain

logger = get_logger(__file__)


with set_configs():
    configs.model.domain = 'referit3d'
    configs.model.scene_graph = '3d'
    configs.model.use_predefined_ccg = False
    configs.train.refexp_add_supervision = True
    configs.train.attrcls_add_supervision = True
    configs.model.sg_dims = [None, 607, 128, 128]
    configs.model.vse_hidden_dims = [None, 607, 128, 128 * 3]
    configs.model.output_dim = 128


def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)


class Model(LeftModel):
    def __init__(self, parsed_train_path, parsed_test_path, idx_to_class_path):
        self.parsed_train_path = parsed_train_path
        self.parsed_test_path = parsed_test_path

        logger.critical('Train parsing: ' + self.parsed_train_path)
        logger.critical('Test parsing: ' + self.parsed_test_path)

        domain = make_domain(self.parsed_test_path)

        super().__init__(domain)

        train_utterance_to_parsed_dict = io.load_pkl(self.parsed_train_path)
        test_utterance_to_parsed_dict = io.load_pkl(self.parsed_test_path)

        utterance_to_parsed_dict = train_utterance_to_parsed_dict.copy()
        utterance_to_parsed_dict.update(test_utterance_to_parsed_dict)
        self.utterance_to_parsed_dict = utterance_to_parsed_dict

        self.attribute_concepts.sort()
        logger.critical('Num attribute concepts: ' + str(len(self.attribute_concepts)))
        k = self.attribute_concepts
        v = list(range(len(self.attribute_concepts)))
        self.attribute_class_to_idx = dict(zip(k, v))

        self.idx_to_class = io.load_pkl(idx_to_class_path)

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        f_sng = self.forward_sng(feed_dict)
        results, executions, parsings, scored = list(), list(), list(), list()

        for i in range(len(feed_dict.input_str)):
            context_size = feed_dict.input_objects_length[i]
            attribute = f_sng[i]['attribute'][:context_size, :]
            relation = f_sng[i]['relation'][:context_size, :context_size, :]
            multi_relation = f_sng[i]['multi_relation'][:context_size, :context_size, :]
            trimmed_f_sng = {
                'attribute': attribute,
                'relation': relation,
                'multi_relation': multi_relation
            }
            input_objects_class = feed_dict.input_objects_class[i][:context_size]

            with self.executor.with_grounding(self.grounding_cls(trimmed_f_sng, self, self.training, self.attribute_class_to_idx, input_objects_class)):
                this_input_str = feed_dict.input_str[i].replace(',', '')

                parsing_list = tuple(self.utterance_to_parsed_dict[this_input_str])
                parsing = self.parser.parse_expression(parsing_list[0])
                execution = self.executor.execute(parsing).tensor
                
                program = execution
                results.append((parsing, program, execution))
                executions.append(execution)
                parsings.append(parsing)
                scored.append(1)

        outputs['parsing'] = parsings
        outputs['results'] = results
        outputs['executions'] = executions
        outputs['scored'] = scored
        
        update_from_loss_module(monitors, outputs, self.refexp_loss(executions, feed_dict.output_target, feed_dict.input_objects_length))
        update_from_loss_module(monitors, outputs, self.attrcls_loss(feed_dict, f_sng, self.attribute_class_to_idx, self.idx_to_class))

        if self.training:
            if configs.train.attrcls_add_supervision:
                return monitors['loss/refexp'] + monitors['loss/attrcls'], monitors, outputs
            else:
                return monitors['loss/refexp'], monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs

    def extract_concepts(self, domain):
        from left.domain import read_concepts_v1
        _, arity_2, arity_3 = read_concepts_v1(domain)
        
        from left.data.referit3d.sr3d_constants import attribute_concepts, view_concepts
        arity_1 = attribute_concepts
        arity_3 += view_concepts

        return arity_1, arity_2, arity_3

    def forward_sng(self, feed_dict):
        f_scene = feed_dict.scene
        f_sng = self.scene_graph(f_scene, feed_dict.input_objects, feed_dict.input_objects_length)
        f_sng = [
            {'attribute': sng[1], 'relation': sng[2], 'multi_relation': sng[3]}
            for sng in f_sng
        ]
        return f_sng


def make_model(parsed_train_path, parsed_test_path, idx_to_class_path):
    return Model(parsed_train_path, parsed_test_path, idx_to_class_path)