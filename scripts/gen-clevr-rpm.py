#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-clevr-rpm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/14/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

import jacinle
import itertools
import random
import os.path as osp
from collections import defaultdict

from concepts.benchmark.clevr.clevr_constants import g_attribute_concepts
from jacinle.cli.argument import JacArgumentParser

parser = JacArgumentParser()
parser.add_argument('--scenes-json', required=True, type='checked_file')
parser.add_argument('--output', required=True)
args = parser.parse_args()

g_concept2attribute = {
    v: k for k, vs in g_attribute_concepts.items() for v in vs
}

g_all_rules = {
    k: list(itertools.combinations_with_replacement(v, 3)) if len(v) < 3 else list(itertools.combinations(v, 3))
    for k, v in g_attribute_concepts.items()
}

print(g_all_rules)


def filter(scene, name, input_):
    if name == 'object':
        return input_
    attribute = g_concept2attribute[name]
    return {i for i in input_ if scene['objects'][i][attribute] == name}


def multi_filter(scene, names, input_):
    for name in names.split():
        input_ = filter(scene, name, input_)
    return input_


def gen_description(rule1_cat, d1, rule2_cat, d2):
    cat_order = ['size', 'color', 'material', 'shape']
    if cat_order.index(rule1_cat) > cat_order.index(rule2_cat):
        rule1_cat, rule2_cat = rule2_cat, rule1_cat
        d1, d2 = d2, d1
    d = d1 + ' ' + d2
    if rule2_cat != 'shape':
        d += ' object'
    if d.startswith('aeiou'):
        d = 'an ' + d
    else:
        d = 'a ' + d
    return d


def main():
    scenes = jacinle.load_json(args.scenes_json)['scenes']

    def find_scene_matching(name, answer):
        for i in range(1000):
            scene_index = random.randint(0, len(scenes) - 1)
            scene = scenes[scene_index]
            res = multi_filter(scene, name, range(len(scene['objects'])))

            if answer is True and len(res) > 0:
                return scene_index, scene
            if answer is False and len(res) == 0:
                return scene_index, scene

    rpms = list()
    for i in range(100):
        rule1_cat, rule2_cat = random.sample(list(g_all_rules.keys()), 2)
        rule1 = random.choice(g_all_rules[rule1_cat])
        rule2 = random.choice(g_all_rules[rule2_cat])
        print(rule1, rule2)

        desired_answer = random.choice([True, False])
        scene_index, scene = find_scene_matching(f'{rule1[2]} {rule2[2]}', desired_answer)

        question = 'There are 9 objects, ordered in a 3x3 grid: '
        for i in range(3):
            for j in range(3):
                if i == 2 and j == 2:
                    continue
                question += f'row {i + 1} col {j + 1} is {gen_description(rule1_cat, rule1[i], rule2_cat, rule2[j])}; '

        question += 'I am missing one object at row 3 col 3. Can you find an object in the scene that can fit there?'

        rpm = {
            'rule1_cat': rule1_cat,
            'rule1': rule1,
            'rule2_cat': rule2_cat,
            'rule2': rule2,
            'answer': desired_answer,
            'scene_index': scene_index,
            'image_filename': scene['image_filename'],
            'target_object': f'{rule1[2]} {rule2[2]}',
            'question': question,
            'program': f'exists(Object, lambda x: {rule1[2]}(x) and {rule2[2]}(x))'
        }
        rpms.append(rpm)

    jacinle.dump_json(args.output, {'rpms': rpms})
    print(f'Saved: "{args.output}".')


if __name__ == '__main__':
    main()

