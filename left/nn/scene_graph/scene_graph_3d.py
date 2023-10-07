#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph_3d.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/22/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""
Scene Graph generation.
"""

import collections
from itertools import product

import torch
import torch.nn as nn

from left.nn.point_net_pp.point_net_pp import PointNetPP, MLP

__all__ = ['SceneGraph3D', 'SceneGraphFeature']


class SceneGraphFeature(collections.namedtuple(
    '_SceneGraphFeature', ('scene_feature', 'object_feature', 'relation_feature', 'multi_relation_feature')
)):
    pass


class SceneGraph3D(nn.Module):
    training: bool

    def __init__(self, pn_output_dim: int, n_obj_classes: int):
        """The default scene graph generator for 3D point clouds."""
        super().__init__()

        self.single_object_encoder = self.single_object_encoder_init(pn_output_dim)
        self.single_object_encoder_for_relation = self.single_object_sparse_encoder_init(pn_output_dim)
        self.object_mlp = MLP(pn_output_dim, [128, 256, n_obj_classes], dropout_rate=0.35)

        self.relation_dense_layer = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim * 1 * 2, pn_output_dim))
        self.relation_dense_layer_2 = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim, pn_output_dim))

        self.multi_relation_dense_layer = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim, pn_output_dim))
        self.multi_relation_dense_layer_2 = nn.Sequential(nn.ReLU(True), nn.Linear(pn_output_dim, pn_output_dim))

    single_object_encoder: PointNetPP
    single_object_encoder_for_relation: PointNetPP
    relation_dense_layer: nn.Sequential
    relation_dense_layer_2: nn.Sequential
    multi_relation_dense_layer: nn.Sequential
    multi_relation_dense_layer_2: nn.Sequential

    def single_object_encoder_init(self, out_dim: int) -> PointNetPP:
        """The default PointNet++ encoder for a 3D object.

        Args:
            out_dims: the dimension of each object feature
        """
        return PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[
                [3, 64, 64, 128],
                [128, 128, 128, 256],
                [256, 256, 512, out_dim]
            ]
        )

    def single_object_sparse_encoder_init(self, out_dim: int) -> PointNetPP:
        """The default PointNet++ encoder for a 3D object.

        Args:
            out_dims: the dimension of each object feature
        """
        return PointNetPP(
            sa_n_points=[8, None],
            sa_n_samples=[8, None],
            sa_radii=[0.4, None],
            sa_mlps=[[3, 32, 32, 64], [64, 64, 64, out_dim]])

    def union(self, cloud1, cloud2):
        return torch.concat([cloud1, cloud2], dim=1)

    def mask_for_location_only(self, target):
        masked_channels = torch.zeros((target.size(0), target.size(1), 3))
        # x, y, z, color
        masked_full = torch.concat([target[:, :, :3], masked_channels.cuda()], dim=-1)
        return masked_full

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(
        self,
        input: torch.Tensor,
        objects: torch.Tensor,
        objects_length: torch.Tensor
    ):
        objects_length = objects_length.detach().cpu()

        # For relation embedding
        pn_object_feat = []
        for i in range(objects.size(1)):
            masked_object = self.mask_for_location_only(objects[:, i, :, :])
            curr_single_obj_feat = self.single_object_encoder_for_relation(masked_object)
            pn_object_feat.append(curr_single_obj_feat)
        pn_object_feat = torch.stack(pn_object_feat)

        relation_feat = [[0 for i in range(objects.size(1))] for j in range(objects.size(1))]
        multi_relation_feat = [[0 for i in range(objects.size(1))] for j in range(objects.size(1))]
        for i, j in product(range(objects.size(1)), range(objects.size(1))):
            union_object_feat = self.union(pn_object_feat[i], pn_object_feat[j])
            union_object_feat = self.relation_dense_layer(union_object_feat)
            union_object_feat = self.relation_dense_layer_2(union_object_feat)
            relation_feat[i][j] = union_object_feat

            multi_union_object_feat = self.multi_relation_dense_layer(union_object_feat)
            multi_union_object_feat = self.multi_relation_dense_layer_2(multi_union_object_feat)
            multi_relation_feat[i][j] = multi_union_object_feat

        relation_for_o = []
        for o in relation_feat:
            relation_for_o.append(torch.stack(o))
        relation_feat = torch.stack(relation_for_o)

        multi_relation_for_o = []
        for o in multi_relation_feat:
            multi_relation_for_o.append(torch.stack(o))
        multi_relation_feat = torch.stack(multi_relation_for_o)

        # For object embedding
        object_feat = []
        for i in range(objects.size(1)):  # Potentially more efficient implementation
            curr_single_obj_feat = self.single_object_encoder(objects[:, i, :, :])
            curr_single_obj_feat = self.object_mlp(curr_single_obj_feat)
            object_feat.append(curr_single_obj_feat)
        object_feat = torch.stack(object_feat)

        outputs = list()
        for i in range(objects.size(0)):
            this_object_feat = object_feat[:, i, :]
            this_relation_feat = relation_feat[:, :, i, :]
            this_multi_relation_feat = multi_relation_feat[:, :, i, :]
            outputs.append(SceneGraphFeature(None, this_object_feat, this_relation_feat, this_multi_relation_feat))

        return outputs

