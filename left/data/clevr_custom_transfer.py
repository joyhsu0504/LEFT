#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : clevr_custom_transfer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/14/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

from concepts.benchmark.clevr.dataset import make_custom_transfer_dataset

g_query_list_keys = {
    'ref': 'refexps',
    'puzzle': 'puzzles',
    'rpm': 'rpms'
}


def make_dataset(mode, scenes_json, questions_json, image_root, output_vocab_json):
    return make_custom_transfer_dataset(
        scenes_json, questions_json, image_root=image_root, output_vocab_json=output_vocab_json,
        query_list_key=g_query_list_keys[mode],
        custom_fields=[],
        incl_scene=False
    )

