#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_left.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Union, List, Dict

from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.config.environ_v2 import configs, set_configs
from concepts.benchmark.clevr.clevr_constants import g_attribute_concepts, g_relational_concepts
from left.models.model import LeftModel
from left.models.losses import CLEVRConceptSupervisionLoss

logger = get_logger(__file__)


with set_configs():
    configs.model.domain = 'clevr'
    configs.model.scene_graph = '2d'
    configs.model.concept_embedding = 'linear'
    configs.train.refexp_add_supervision = True
    configs.train.attrcls_add_supervision = False
    configs.train.concept_add_supervision = False


_g_strict = True


def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)


class ExecutionFailed(Exception):
    pass


class Model(LeftModel):
    def __init__(self, domain, parses: Dict[str, Union[str, List[str]]], output_vocab, custom_transfer=None):
        super().__init__(domain, output_vocab)
        self.parses = parses

        self.concept_supervision_loss = CLEVRConceptSupervisionLoss(
            g_attribute_concepts, g_relational_concepts, configs.train.concept_add_supervision
        )
        self.custom_transfer = custom_transfer

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        f_sng = self.forward_sng(feed_dict)
        outputs['results'] = list()
        outputs['groundings'] = list()
        outputs['executions'] = list()
        outputs['parsings'] = list()
        outputs['execution_traces'] = list()

        for i in range(len(feed_dict.question_raw)):
            context_size = feed_dict.objects_length[i]
            trimmed_f_sng = {
                'attribute': f_sng[i]['attribute'][:context_size, :],
                'relation': f_sng[i]['relation'][:context_size, :context_size, :]
            }

            grounding = self.grounding_cls(trimmed_f_sng, self, self.training, apply_relation_mask=True, attribute_concepts={k.capitalize(): v for k, v in g_attribute_concepts.items()})
            outputs['groundings'].append(grounding)
            with self.executor.with_grounding(grounding):
                question = feed_dict.question_raw[i]
                raw_parsing = self.parses[question]
                if isinstance(raw_parsing, list):
                    raw_parsing = raw_parsing[0]

                parsing, program, execution, trace = None, None, None, None
                try:
                    try:
                        parsing = self.parser.parse_expression(raw_parsing)
                        program = parsing
                        # print(repr(program))
                    except Exception as e:
                        raise ExecutionFailed('Parsing failed for question: {}.'.format(question)) from e

                    try:
                        if not self.training:
                            with self.executor.record_execution_trace() as trace_getter:
                                execution = self.executor.execute(program)
                                trace = trace_getter.get()
                        else:
                            execution = self.executor.execute(program)
                    except (KeyError, AttributeError) as e:
                        logger.exception('Execution failed for question: {}\nProgram: {}.'.format(question, program))
                        raise ExecutionFailed('Execution failed for question: {}\nProgram: {}.'.format(question, program)) from e
                except ExecutionFailed as e:
                    if _g_strict is True:
                        raise e

            outputs['results'].append((parsing, program, execution))
            outputs['executions'].append(execution)
            outputs['parsings'].append(parsing)
            outputs['execution_traces'].append(trace)

        if not self.custom_transfer:
            update_from_loss_module(monitors, outputs, self.qa_loss(outputs['executions'], feed_dict.answer, feed_dict.question_type))
            update_from_loss_module(monitors, outputs, self.concept_supervision_loss(outputs['groundings'], feed_dict))
        else:
            raise NotImplementedError()

        if self.training:
            loss = monitors['loss/qa']
            if configs.train.concept_add_supervision:
                loss += monitors['loss/concept_supervision']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs

    def forward_sng(self, feed_dict):
        f_scene = self.resnet(feed_dict.image)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)
        f_sng = [{'attribute': sng[1], 'relation': sng[2]} for sng in f_sng]
        return f_sng


def make_model(args, domain, parses, output_vocab):
    return Model(domain, parses, output_vocab)

