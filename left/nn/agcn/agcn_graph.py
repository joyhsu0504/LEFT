#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : agcn_graph.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 03/24/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

# adopted from https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/graph/ntu_rgb_d.py
# and https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/graph/tools.py
import sys
import numpy as np

num_node = 22
self_link = [(i, i) for i in range(num_node)]
# NTU RGB+D Mapping
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
# helpful images for translating: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf,
# https://media.arxiv-vanity.com/render-output/6081457/x9.png

# zero index
inward_ori_index = [(0, 3), (3, 6), (9, 6), (12, 9), (15, 12), (13, 12), (16, 13), (18, 16),
                    (20, 18), (14, 12), (17, 14), (19, 17), (21, 19), (1, 0),
                    (4, 1), (7, 4), (10, 7), (2, 0), (5, 2), (8, 5), (11, 8)]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class AGCNGraph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

