#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/18/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

import collections
import torch
import torch.nn.functional as F
import jacinle
import jactorch.nn as jacnn
import numpy as np

from typing import Optional, Sequence, List, Dict

from concepts.benchmark.common.vocab import Vocab
from left.nn.losses import MultitaskLossBase
from left.models.reasoning.reasoning import NCOneTimeComputingGrounding

__all__ = ['RefExpLoss', 'AttrClsLoss', 'QALoss', 'PickPlaceLoss']


class RefExpLoss(jacnn.TorchApplyRecorderMixin, MultitaskLossBase):
    def __init__(self, add_supervision=True, softmax=False, one_hot=False, context_objects_only=False):
        super().__init__()
        self.add_supervision = add_supervision
        self.softmax = softmax
        self.one_hot = one_hot

    def forward(self, input, target, input_objects_length):
        monitors = dict()
        outputs = dict()

        batch_size = len(input)
        loss, acc, acc_instance = [], [], []
        for i in range(batch_size):
            this_input = input[i]
            this_target = target[i]

            # Softmax
            if this_input.size() == torch.Size([]):
                continue
            if this_input.size(0) <= this_target:
                continue
            try:
                this_loss = self._batched_xent_loss(this_input, this_target)
                a = float(torch.argmax(this_input) == this_target)
                ai = (this_input[this_target] > 0).float()

                loss.append(this_loss)
                acc.append(a)
                acc_instance.append(ai)
            # TODO(Jiayuan Mao @ 2023/04/05): document what's this except thing trying to catch...
            except:
                continue

        avg_loss = sum(loss) / len(loss) if len(loss) != 0 else 0.0
        avg_acc = sum(acc) / len(acc) if len(acc) != 0 else 0.0
        avg_ai = sum(acc_instance) / len(acc_instance) if len(acc_instance) != 0 else 0.0
        if self.training and self.add_supervision:
            monitors['loss/refexp'] = avg_loss

        if self.training:
            monitors['acc/refexp'] = avg_acc
            monitors['acc/refexp/instance'] = avg_ai
        else:
            monitors['validation/acc/refexp'] = avg_acc
            monitors['validation/acc/refexp/instance'] = avg_ai

        return monitors, outputs


class AttrClsLoss(MultitaskLossBase):
    def __init__(self, add_supervision=False):
        super().__init__()
        self.add_supervision = add_supervision

    def forward(self, feed_dict, f_sng, attribute_class_to_idx, idx_to_class):
        outputs, monitors = dict(), dict()

        objects = [f['attribute'] for f in f_sng]
        all_f = torch.stack(objects)
        object_labels = feed_dict['input_objects_class']

        all_scores = []
        attribute_concepts = list(attribute_class_to_idx.keys())
        attribute_concepts.sort()
        for concept in attribute_concepts:
            this_score = all_f[:, :, attribute_class_to_idx[concept]]
            all_scores.append(this_score)
        all_scores = torch.stack(all_scores, dim=-1)

        accs, losses = [], []
        concepts_to_accs, concepts_to_pred_concepts = [], []
        for b in range(object_labels.size(0)):
            gt_concepts_to_accs = collections.defaultdict(list)
            gt_concepts_to_pred_concepts = collections.defaultdict(list)
            for i in range(object_labels.size(1)):
                gt_label = int(object_labels[b, i].cpu().numpy())
                gt_class = idx_to_class[gt_label].replace(' ', '_') + '_Object'

                if gt_class in attribute_concepts:
                    gt_class_index = attribute_concepts.index(gt_class)
                else:
                    continue

                pred_scores_for_object = all_scores[b, i, :]
                pred_max_class_index = int(torch.argmax(pred_scores_for_object).cpu().numpy())
                pred_class = attribute_concepts[pred_max_class_index]
                gt_concepts_to_pred_concepts[gt_class].append(pred_class)

                this_acc = float(pred_max_class_index == gt_class_index)
                accs.append(this_acc)
                gt_concepts_to_accs[gt_class].append(this_acc)

                this_loss = self._sigmoid_xent_loss(pred_scores_for_object, torch.tensor(gt_class_index).cuda())
                losses.append(this_loss)

            concepts_to_accs.append(gt_concepts_to_accs)
            concepts_to_pred_concepts.append(gt_concepts_to_pred_concepts)

        avg_acc = sum(accs) / len(accs) if len(accs) != 0 else 0.0
        avg_loss = sum(losses) / len(losses) if len(losses) != 0 else 0.0

        outputs['concepts_to_accs'] = concepts_to_accs
        outputs['concepts_to_pred_concepts'] = concepts_to_pred_concepts

        if self.training and self.add_supervision:
            monitors['loss/attrcls'] = avg_loss

        if self.training:
            monitors['acc/attrcls'] = avg_acc
            monitors['train/acc/attrcls'] = avg_acc
        else:
            monitors['validation/acc/attrcls'] = avg_acc

        return monitors, outputs


class QALoss(MultitaskLossBase):
    def __init__(self, output_vocab: Optional[Vocab] = None, add_supervision: bool = True):
        super().__init__()
        self.output_vocab = output_vocab
        self.add_supervision = add_supervision

    def forward(self, execution_results, groundtruth, question_types):
        monitors, outputs = collections.defaultdict(list), dict()
        outputs['pred_answers'] = list()

        assert len(execution_results) == len(groundtruth)
        for result, gt_answer, question_type in zip(execution_results, groundtruth, question_types):
            if result is None:
                monitors['acc/success_exec'].append(0.0)
                monitors['acc/qa'].append(0.0)
                continue

            try:
                result_typename = result.dtype.typename
            # TODO(Joy Hsu @ 2023/10/04): remove exceptions for tensor result type.
            except:
                result_typename = result.dtype
            if result_typename == 'bool':
                pred_answer = 'yes' if result.tensor.item() > 0 else 'no'
            elif result_typename == 'int64':
                pred_answer = str(int(result.tensor.round().item()))
            else:
                try:
                    pred_answer = self.output_vocab.idx2word[result.tensor.argmax().item()]
                except:
                    pred_answer = self.output_vocab.idx2word[result.argmax().item()]
            outputs['pred_answers'].append(pred_answer)

            if result_typename == 'bool':
                if isinstance(gt_answer, bool):
                    this_loss = self._bce_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_accuracy = bool(result.tensor.item() > 0) == gt_answer
                    # print(f'compute loss pred={result.tensor.sigmoid().item()} gt={gt_answer} loss={this_loss.item()}')
                else:
                    this_loss, this_accuracy = 10, False
            elif result_typename == 'int64':
                if isinstance(gt_answer, int):
                    this_loss = self._mse_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_accuracy = pred_answer == str(gt_answer)
                else:
                    this_loss, this_accuracy = 10, False
            elif result_typename == 'Object':
                this_loss, this_accuracy = 10, False
            else:
                if isinstance(gt_answer, str):
                    try:
                        this_loss = self._xent_loss(result.tensor, torch.tensor(self.output_vocab.word2idx[gt_answer]).to(result.tensor.device))
                    except:
                        this_loss = self._xent_loss(result, torch.tensor(self.output_vocab.word2idx[gt_answer]).to(result.device))
                    this_accuracy = pred_answer == gt_answer
                else:
                    this_loss, this_accuracy = 10, False

            if self.training and self.add_supervision:
                monitors['loss/qa'].append(this_loss)

            monitors['acc/success_exec'].append(1.0)
            monitors['acc/qa'].append(this_accuracy)
            monitors['acc/qa_succ_exec'].append(this_accuracy)
            monitors[f'acc/qa/{question_type}'].append(this_accuracy)

        for k, vs in list(monitors.items()):
            monitors[k + '/n'] = len(vs)
            monitors[k] = sum(vs) / len(vs) if len(vs) != 0 else 0.0

        return monitors, outputs


class CLEVRConceptSupervisionLoss(MultitaskLossBase):
    def __init__(self, attribute_concepts: Dict[str, List[str]], relational_concepts: Dict[str, List[str]], add_supervision: bool = False):
        super().__init__()
        self.add_supervision = add_supervision
        self.attribute_concepts = attribute_concepts
        self.relational_concepts = relational_concepts

    def forward(self, groundings: Sequence[NCOneTimeComputingGrounding], feed_dict: jacinle.GView):
        monitors, outputs = collections.defaultdict(list), dict()

        attribute_classification_scores = collections.defaultdict(list)
        attribute_relation_scores = collections.defaultdict(list)
        relation_scores = collections.defaultdict(list)

        for grounding in groundings:
            for attr_name, concepts in self.attribute_concepts.items():
                this_attribute_classification = grounding.compute_similarities_batch('attribute', [f'{c}_Object' for c in concepts])
                this_attribute_relation = grounding.compute_similarity('relation', f'same_{attr_name}_Object_Object')

                attribute_classification_scores[attr_name].append(this_attribute_classification)
                attribute_relation_scores[attr_name].append(this_attribute_relation.reshape(-1))

            for attr_name, concepts in self.relational_concepts.items():
                this_relation = grounding.compute_similarities_batch('relation', [f'{c}_Object_Object' for c in concepts])
                relation_scores[attr_name].append(this_relation.reshape(-1, this_relation.shape[-1]))

        for k, v in attribute_classification_scores.items():
            attribute_classification_scores[k] = torch.concat(v, dim=0)
        for k, v in attribute_relation_scores.items():
            attribute_relation_scores[k] = torch.concat(v, dim=0)
        for k, v in relation_scores.items():
            relation_scores[k] = torch.concat(v, dim=0)

        loss = 0
        for k, v in attribute_classification_scores.items():
            if f'attribute_{k}' not in feed_dict:
                continue
            accuracy = (v.argmax(dim=-1) == feed_dict[f'attribute_{k}']).float().mean()
            monitors[f'acc/attrcls/{k}'] = accuracy.item()
            monitors[f'acc/attrcls/{k}/n'] = v.shape[0]
            if self.training and self.add_supervision:
                monitors[f'loss/attrcls/{k}'] = self._sigmoid_xent_loss(v, feed_dict[f'attribute_{k}']).mean()
                monitors[f'loss/attrcls/{k}/n'] = v.shape[0]
                loss += monitors[f'loss/attrcls/{k}']
        for k, v in attribute_relation_scores.items():
            if f'attribute_relation_{k}' not in feed_dict:
                continue
            accuracy = ((v > 0) == feed_dict[f'attribute_relation_{k}']).float().mean()
            monitors[f'acc/attrrel/{k}'] = accuracy.item()
            monitors[f'acc/attrrel/{k}/n'] = v.shape[0]
            if self.training and self.add_supervision:
                monitors[f'loss/attrrel/{k}'] = self._bce_loss(v, feed_dict[f'attribute_relation_{k}'].float()).mean()
                monitors[f'loss/attrrel/{k}/n'] = v.shape[0]
                loss += monitors[f'loss/attrrel/{k}']
        for k, v in relation_scores.items():
            if f'relation_{k}' not in feed_dict:
                continue
            accuracy = ((v > 0) == feed_dict[f'relation_{k}']).float().mean()
            monitors[f'acc/rel/{k}'] = accuracy.item()
            monitors[f'acc/rel/{k}/n'] = v.shape[0]
            if self.training and self.add_supervision:
                monitors[f'loss/rel/{k}'] = self._bce_loss(v, feed_dict[f'relation_{k}']).mean()
                monitors[f'loss/rel/{k}/n'] = v.shape[0]
                loss += monitors[f'loss/rel/{k}']

        if self.training and self.add_supervision:
            monitors['loss/concept_supervision'] = loss

        return monitors, outputs


class PickPlaceLoss(MultitaskLossBase):
    def __init__(self):
        super().__init__()

    def cross_entropy_with_logits(self, pred, labels, reduction='sum'):
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def forward(self, execution_results, inp_img, p0, p0_theta, p1, p1_theta, pick_attn, goal_transport):
        assert len(execution_results)==1 and len(inp_img)==1 # Assume batch size 1
        outputs, monitors = dict(), dict()

        # Pick loss
        pick_loc, place_output = execution_results[0]
        inp_img = np.array(inp_img[0])
        gt_pick_center = np.array(p0[0])
        gt_pick_theta = np.array(p0_theta[0])

        # Get label.
        theta_i = gt_pick_theta / (2 * np.pi / pick_attn.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % pick_attn.n_rotations

        label_size = inp_img.shape[:2] + (pick_attn.n_rotations,)
        label = np.zeros(label_size)
        label[gt_pick_center[0], gt_pick_center[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=pick_loc.device)

        pick_loc = pick_loc.permute(2, 0, 1)
        pick_loc = pick_loc.reshape(1, np.prod(pick_loc.shape))

        # Get loss.
        pick_loss = self.cross_entropy_with_logits(pick_loc, label)


        # Place loss
        gt_place_center = np.array(p1[0])
        gt_place_theta = np.array(p1_theta[0])

        itheta = gt_place_theta / (2 * np.pi / goal_transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % goal_transport.n_rotations

        # Get one-hot pixel label map.
        label_size = inp_img.shape[:2] + (goal_transport.n_rotations,)
        label = np.zeros(label_size)
        label[gt_place_center[0], gt_place_center[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=place_output.device)
        place_output = place_output.reshape(1, np.prod(place_output.shape))

        place_loss = self.cross_entropy_with_logits(place_output, label)
        goal_transport.iters += 1 # Check

        loss = pick_loss + place_loss

        if self.training:
            monitors['loss/pickplace'] = loss

        return monitors, outputs
