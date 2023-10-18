#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : model.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.


import torch.nn as nn
import jactorch.nn as jacnn

from typing import Any, Optional, Tuple, List, Dict

from jacinle.config.environ_v2 import configs, def_configs_func
from jacinle.logging import get_logger

from concepts.dsl.function_domain import FunctionDomain
from concepts.benchmark.common.vocab import Vocab
from left.generalized_fol_parser import NCGeneralizedFOLPythonParser
from left.models.concept.concept_embedding import NCVSEConceptEmbedding, NCLinearConceptEmbedding
from left.domain import read_concepts_v2, read_description_categories

logger = get_logger(__file__)

__all__ = ['LeftModel']


class LeftModel(nn.Module):
    @staticmethod
    @def_configs_func
    def _def_configs():
        # model configs for scene graph
        configs.model.domain = 'referit3d'
        configs.model.scene_graph = '3d'
        configs.model.concept_embedding = 'vse'
        configs.model.sg_dims = [None, 128, 128, 128]

        # model configs for visual-semantic embeddings
        configs.model.vse_hidden_dims = [None, 128, 128, 128 * 3]
        configs.model.output_dim = 128

        # supervision configs
        configs.model.use_predefined_ccg = False
        configs.train.refexp_add_supervision = True
        configs.train.attrcls_add_supervision = False
        configs.train.concept_add_supervision = False
        configs.train.weight_decay = 0

        return configs

    def __init__(self, domain, output_vocab: Optional[Vocab] = None):
        super().__init__()
        self._def_configs()

        self.domain = domain
        self.attribute_concepts, self.relational_concepts, self.multi_relational_concepts = self.extract_concepts(self.domain)
        self.attribute_description_categories = self.extract_description_categories(self.domain)

        self.output_vocab = output_vocab
        self.description_vocab_size = len(self.output_vocab) if self.output_vocab is not None else None

        # TODO(Jiayuan Mao @ 2023/03/18): probably we shuold just move the scene_graph generation part out.
        # I think, for now, we will anyway be using different scene graph generation methods/interfaces for different datasets...
        self.use_resnet = False
        if configs.model.scene_graph == '2d':
            import left.nn.scene_graph.scene_graph_2d as sng
            self.scene_graph = sng.SceneGraph2D(256, configs.model.sg_dims, 16)

            import jactorch.models.vision.resnet as resnet
            self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
            self.resnet.layer4 = jacnn.Identity()
            self.use_resnet = True
        elif configs.model.scene_graph == '3d':
            import left.nn.scene_graph.scene_graph_3d as sng
            self.scene_graph = sng.SceneGraph3D(configs.model.output_dim, len(self.attribute_concepts))
        elif configs.model.scene_graph == 'skeleton':
            import left.nn.scene_graph.scene_graph_skeleton as sng
            self.scene_graph = sng.SceneGraphSkeleton(len(self.attribute_concepts), self.description_vocab_size)
        elif configs.model.scene_graph is None:
            self.scene_graph = None
        else:
            raise ValueError(f'Unknown scene graph type: {configs.model.scene_graph}.')

        if configs.model.concept_embedding == 'vse':
            self.attribute_embedding = NCVSEConceptEmbedding()
            self.relation_embedding = NCVSEConceptEmbedding()
            self.multi_relation_embedding = NCVSEConceptEmbedding()
            from left.models.reasoning.reasoning import LeftGrounding
            self.grounding_cls = LeftGrounding
        elif configs.model.concept_embedding == 'linear':
            self.attribute_embedding = NCLinearConceptEmbedding()
            self.relation_embedding = NCLinearConceptEmbedding()
            self.multi_relation_embedding = NCLinearConceptEmbedding()
            from left.models.reasoning.reasoning import NCOneTimeComputingGrounding
            self.grounding_cls = NCOneTimeComputingGrounding
        elif configs.model.concept_embedding == 'linear-tied-attr':
            self.attribute_embedding = NCLinearConceptEmbedding(tied_attributes=True)
            self.relation_embedding = NCLinearConceptEmbedding()
            self.multi_relation_embedding = NCLinearConceptEmbedding()
            from left.models.reasoning.reasoning import NCOneTimeComputingGrounding
            self.grounding_cls = NCOneTimeComputingGrounding
        elif configs.model.concept_embedding == 'clip':
            self.attribute_embedding = NCVSEConceptEmbedding()
            self.relation_embedding = NCVSEConceptEmbedding()
            self.multi_relation_embedding = NCVSEConceptEmbedding()
            from left.models.reasoning.reasoning import NCDenseClipGrounding
            self.grounding_cls = NCDenseClipGrounding
        else:
            raise ValueError(f'Unknown concept embedding type: {configs.model.concept_embedding}.')

        self.init_concept_embeddings()

        from left.generalized_fol_executor import NCGeneralizedFOLExecutor
        self.parser = NCGeneralizedFOLPythonParser(self.domain, inplace_definition=False, inplace_polymorphic_function=True, inplace_definition_type=False)
        self.executor = NCGeneralizedFOLExecutor(self.domain, self.parser)

        from left.models.losses import RefExpLoss, AttrClsLoss, QALoss, PickPlaceLoss
        self.refexp_loss = RefExpLoss(add_supervision=configs.train.refexp_add_supervision)
        self.attrcls_loss = AttrClsLoss(add_supervision=configs.train.attrcls_add_supervision)
        self.qa_loss = QALoss(output_vocab)
        self.pickplace_loss = PickPlaceLoss()

    def extract_concepts(self, domain: FunctionDomain) -> Tuple[List[str], List[str], List[str]]:
        return read_concepts_v2(domain)

    def extract_description_categories(self, domain: FunctionDomain) -> List[str]:
        return read_description_categories(domain)

    def init_concept_embeddings(self):
        if configs.model.concept_embedding == 'vse':
            for arity, src, tgt in zip(
                [1, 2, 3],
                [self.attribute_concepts, self.relational_concepts, self.multi_relational_concepts],
                [self.attribute_embedding, self.relation_embedding, self.multi_relation_embedding]
            ):
                tgt.init_attribute('all', configs.model.sg_dims[arity])
                for word in src:
                    tgt.init_concept(word, configs.model.vse_hidden_dims[arity], 'all')
        elif configs.model.concept_embedding in ('linear', 'linear-tied-attr'):
            for arity, src, tgt in zip(
                [1, 2, 3],
                [self.attribute_concepts, self.relational_concepts, self.multi_relational_concepts],
                [self.attribute_embedding, self.relation_embedding, self.multi_relation_embedding]
            ):
                for word in src:
                    tgt.init_concept(word, configs.model.sg_dims[arity])
            if len(self.attribute_concepts) > 0:
                if self.description_vocab_size is not None:
                    for word in self.attribute_description_categories:
                        self.attribute_embedding.init_attribute(word, configs.model.sg_dims[1], self.description_vocab_size)
            for tgt in [self.attribute_embedding, self.relation_embedding, self.multi_relation_embedding]:
                tgt.init_linear_layers()
        elif configs.model.concept_embedding == 'clip':
            pass
        else:
            raise ValueError(f'Unknown concept embedding type: {configs.model.concept_embedding}.')

    def forward_sng(self, feed_dict):
        raise NotImplementedError()

    def execute_program_from_parsing_string(self, question: str, raw_parsing: str, grounding, outputs: Dict[str, Any]):
        parsing, program, execution, trace = None, None, None, None
        with self.executor.with_grounding(grounding):
            try:
                try:
                    parsing = raw_parsing
                    program = self.parser.parse_expression(raw_parsing)
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
                print(e)

        outputs.setdefault('results', list()).append((parsing, program, execution))
        outputs.setdefault('executions', list()).append(execution)
        outputs.setdefault('parsings', list()).append(parsing)
        outputs.setdefault('execution_traces', list()).append(trace)


class ExecutionFailed(Exception):
    pass


