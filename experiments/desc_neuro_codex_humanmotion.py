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
    configs.model.domain = 'humanmotion'
    configs.model.scene_graph = 'skeleton'
    configs.model.use_predefined_ccg = False
    configs.train.refexp_add_supervision = True
    configs.train.attrcls_add_supervision = True
    configs.model.vse_hidden_dims = [None, 128, 224, 128 * 3]


def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)


class Model(LeftModel):
    def __init__(self, parsed_train_path, parsed_test_path, output_vocab):
        self.parsed_train_path = parsed_train_path
        self.parsed_test_path = parsed_test_path

        logger.critical('Train parsing: ' + self.parsed_train_path)
        logger.critical('Test parsing: ' + self.parsed_test_path)

        domain = make_domain(self.parsed_test_path)

        super().__init__(domain, output_vocab)
        
        from left.generalized_fol_executor import NCGeneralizedFOLExecutor
        self.executor = NCGeneralizedFOLExecutor(self.domain, self.parser, allow_shift_grounding=True)

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

    @functools.lru_cache(maxsize=None, typed=False)
    def get_legal(self, parsing_list):  # Potentially move to model.py
        legal_list = []
        for raw_parsing in list(parsing_list):
            try:
                legal_parsing = self.parser.parse_expression(raw_parsing)
                legal_execution = self.executor.execute(legal_parsing).tensor.detach()
                legal_list.append((legal_parsing, legal_execution))
            except:
                continue
        return legal_list

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        f_sng = self.forward_sng(feed_dict)
        results = list()
        executions = list()
        parsings = list()
        scored, parsed, parsed_legal, execution_legal = list(), list(), list(), list()
        
        for i in range(len(feed_dict.program_tree)):
            with self.executor.with_grounding(self.grounding_cls(f_sng[i], self, self.training, self.attribute_class_to_idx, None)):
                this_input_str = feed_dict.question_text[i]
                
                parsing_list = tuple([self.utterance_to_parsed_dict[this_input_str]])
                parsing = self.parser.parse_expression(parsing_list[0])
                execution = self.executor.execute(parsing).tensor
                
                program = execution
                results.append((parsing, program, execution))
                executions.append(execution)
                parsings.append(parsing)
                scored.append(1)

                # TODO: remove or update
                parsed.append(0)
                parsed_legal.append(0)
                execution_legal.append(0)

        outputs['parsing'] = parsings
        outputs['results'] = results
        outputs['executions'] = executions
        outputs['scored'] = scored
        outputs['legality'] = (parsed, parsed_legal, execution_legal)
        
        update_from_loss_module(monitors, outputs, self.qa_loss(outputs['executions'], feed_dict.answer, feed_dict.question_type))
        
        if self.training:
            loss = monitors['loss/qa']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs

    def extract_concepts(self, domain):
        from left.domain import read_concepts_v2
        _, arity_2, arity_3 = read_concepts_v2(domain)
        
        from concepts.benchmark.vision_language.babel_qa.humanmotion_constants import attribute_concepts_mapping
        arity_1 = attribute_concepts_mapping['Motion'] + attribute_concepts_mapping['Part'] + attribute_concepts_mapping['Direction']

        return arity_1, arity_2, arity_3

    def forward_sng(self, feed_dict):
        motion_encodings, motion_encodings_rel, motion_encodings_output_vocab = self.scene_graph(feed_dict.joints)
        f_sng = []
        start_seg = 0
        for seq_num_segs in feed_dict.num_segs:
            f_sng.append({'attribute': motion_encodings[start_seg:start_seg+seq_num_segs],
                          'relation': motion_encodings_rel[start_seg:start_seg+seq_num_segs],
                          'output_vocab': motion_encodings_output_vocab[start_seg:start_seg+seq_num_segs]})
            start_seg += seq_num_segs
        assert start_seg == motion_encodings.size()[0]
        return f_sng


def make_model(parsed_train_path, parsed_test_path, output_vocab):
    return Model(parsed_train_path, parsed_test_path, output_vocab)

