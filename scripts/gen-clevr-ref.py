#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-clevr-ref.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/13/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

import jacinle
import itertools
import random
import os.path as osp

from concepts.benchmark.clevr.clevr_constants import g_attribute_concepts
from jacinle.cli.argument import JacArgumentParser

parser = JacArgumentParser()
parser.add_argument('--scenes-json', required=True, type='checked_file')
parser.add_argument('--output', required=True)
args = parser.parse_args()

g_concept2attribute = {
    v: k for k, vs in g_attribute_concepts.items() for v in vs
}


def filter(scene, name, input_):
    if name == 'object':
        return input_
    attribute = g_concept2attribute[name]
    return {i for i in input_ if scene['objects'][i][attribute] == name}


def multi_filter(scene, names, input_):
    for name in names.split():
        input_ = filter(scene, name, input_)
    return input_


def relate(scene, name, input_):
    if len(input_) != 1:
        raise ValueError()

    input_ = list(input_)[0]
    return set(scene['relationships'][name][input_])


def execute(scene, program, template_slots):
    stack = list()
    for token in program.split():
        if token == 'S':
            stack.append(set(range(len(scene['objects']))))
        elif token == 'AND':
            stack.append(stack.pop() & stack.pop())
        elif token.startswith('OBJ'):
            concept_name = template_slots[token]
            if isinstance(concept_name, int):
                stack.append({concept_name})
            else:
                stack.append(multi_filter(scene, concept_name, stack.pop()))
        elif token.startswith('R'):
            concept_name = template_slots[token]
            stack.append(relate(scene, concept_name, stack.pop()))
        else:
            raise ValueError('Unknown token: {}.'.format(token))

    if len(stack) != 1:
        raise ValueError('Invalid program.')
    if len(stack[0]) != 1:
        raise ValueError('Invalid program.')
    return list(stack[0])[0]


def gen_all_filter_ops():
    for x in itertools.product(
        g_attribute_concepts['size'] + [''],
        g_attribute_concepts['color'] + [''],
        g_attribute_concepts['material'] + [''],
        g_attribute_concepts['shape'] + ['object'],
    ):
        yield ' '.join(x for x in x if x)


def gen_all_relate_ops():
    return ['left', 'right', 'front', 'behind']


g_all_filter_ops = list(gen_all_filter_ops())
g_all_filter_ops.sort(key=lambda x: len(x.split()))
g_all_relate_ops = list(gen_all_relate_ops())


def check_filter_unique(scene, x):
    input_ = range(len(scene['objects']))
    return len(multi_filter(scene, x, input_)) == 1


def gen_filter_string(f, vname):
    return ' and '.join([
        f'{method}({vname})' for method in f.split()
    ])


g_templates_1 = [
    'Select the {OBJ1}.',
    'Find the {OBJ1}.',
]


def ground_program1(scene, unique_filters):
    program = 'S OBJ1'
    sentence_for_x = {}
    for f in unique_filters:
        slot_dict = {'OBJ1': f}
        try:
            obj = execute(scene, program, slot_dict)
        except ValueError:
            continue

        template = random.choice(g_templates_1)
        sentence = template.format(**slot_dict)
        sentence_len = len(sentence.split())
        if obj not in sentence_for_x or sentence_len < sentence_for_x[obj][-1]:
            sentence_for_x[obj] = (sentence, program, slot_dict, obj, sentence_len)

    for sentence, slot_program, slot_dict, obj, _ in sentence_for_x.values():
        obj1_string = gen_filter_string(slot_dict['OBJ1'], 'x')
        program = f'point(Object, lambda x: {obj1_string})'
        yield sentence, program, slot_program, slot_dict, obj


g_templates_2 = [
    'Find the {OBJ1} that is {R1} the {OBJ2}.',
    'Point-to the {OBJ1} that is {R1} the {OBJ2}.',
]


def ground_program2(scene, unique_filters):
    program  = 'S OBJ2 R1 OBJ1'
    program1 = ground_program1(scene, unique_filters)

    sentence_for_x = {}
    for _, _, _, slot_dict1, obj2 in program1:
        for f in g_all_filter_ops:
            for r in g_all_relate_ops:
                slot_dict = {'OBJ1': f, 'OBJ2': obj2, 'R1': r}
                try:
                    obj = execute(scene, 'OBJ2 R1 OBJ1', slot_dict)
                except ValueError:
                    continue

                template = random.choice(g_templates_2)
                slot_dict = {'OBJ1': f, 'OBJ2': slot_dict1['OBJ1'], 'R1': r}
                sentence = template.format(**slot_dict)
                sentence_len = len(sentence.split())
                if obj not in sentence_for_x or sentence_len < sentence_for_x[obj][-1]:
                    sentence_for_x[obj] = (sentence, program, slot_dict, obj, sentence_len)

    for sentence, slot_program, slot_dict, obj, _ in sentence_for_x.values():
        obj1_string = gen_filter_string(slot_dict['OBJ1'], 'x')
        r = slot_dict['R1']
        obj2_string = gen_filter_string(slot_dict['OBJ2'], 'y')
        program = f'point(Object, lambda x: {obj1_string} and {r}(x, iota(Object, lambda y: {obj2_string})))'
        yield sentence, program, slot_program, slot_dict, obj


g_templates_3 = [
    'Point-to the {OBJ1} that is {R1} the {OBJ2} and {R2} the {OBJ3}.',
    'Select the {OBJ1} that is {R1} the {OBJ2} and {R2} the {OBJ3}.'
]


def ground_program3(scene, unique_filters):
    program = 'S OBJ3 R2 S OBJ2 R1 AND OBJ1'
    program1 = ground_program1(scene, unique_filters)

    sentence_for_x = {}
    for _, _, _, slot_dict1, obj2 in program1:
        for _, _, _, slot_dict2, obj3 in program1:
            if obj2 == obj3:
                continue
            for f in g_all_filter_ops:
                for r1 in g_all_relate_ops:
                    for r2 in g_all_relate_ops:
                        slot_dict = {'OBJ1': f, 'OBJ2': obj2, 'R1': r1, 'OBJ3': obj3, 'R2': r2}
                        try:
                            obj = execute(scene, 'OBJ3 R2 OBJ2 R1 AND OBJ1', slot_dict)
                        except ValueError:
                            continue

                        template = random.choice(g_templates_3)
                        slot_dict = {'OBJ1': f, 'OBJ2': slot_dict1['OBJ1'], 'R1': r1, 'OBJ3': slot_dict2['OBJ1'], 'R2': r2}
                        sentence = template.format(**slot_dict)
                        sentence_len = len(sentence.split())
                        if obj not in sentence_for_x or sentence_len < sentence_for_x[obj][-1]:
                            sentence_for_x[obj] = (sentence, program, slot_dict, obj, sentence_len)

    for sentence, slot_program, slot_dict, obj, _ in sentence_for_x.values():
        obj1_string = gen_filter_string(slot_dict['OBJ1'], 'x')
        r1 = slot_dict['R1']
        obj2_string = gen_filter_string(slot_dict['OBJ2'], 'y')
        r2 = slot_dict['R2']
        obj3_string = gen_filter_string(slot_dict['OBJ3'], 'z')
        program = f'point(Object, lambda x: {obj1_string} and {r1}(x, iota(Object, lambda y: {obj2_string})) and {r2}(x, iota(Object, lambda z: {obj3_string})))'
        yield sentence, program, slot_program, slot_dict, obj

    return [x[:-1] for x in sentence_for_x.values()]


g_templates_4 = [
    'There is a {OBJ2} that is {R2} the {OBJ3}, find the {OBJ1} that is {R1} it.',
    'There is a {OBJ2} that is {R2} the {OBJ3}, select the {OBJ1} that is {R1} it.'
]


def ground_program4(scene, unique_filters):
    program = 'S OBJ3 R2 S OBJ2 R1 OBJ1'
    program2 = ground_program2(scene, unique_filters)

    sentence_for_x = {}
    for _, _, _, slot_dict2, obj2 in program2:
        for f in g_all_filter_ops:
            for r1 in g_all_relate_ops:
                slot_dict = {'OBJ1': f, 'R1': r1, 'OBJ2': obj2}
                try:
                    obj = execute(scene, 'OBJ2 R1 OBJ1', slot_dict)
                except ValueError:
                    continue

                template = random.choice(g_templates_4)
                slot_dict = {'OBJ1': f, 'R1': r1, 'OBJ2': slot_dict2['OBJ1'], 'OBJ3': slot_dict2['OBJ2'], 'R2': slot_dict2['R1']}
                sentence = template.format(**slot_dict)
                sentence_len = len(sentence.split())
                if obj not in sentence_for_x or sentence_len < sentence_for_x[obj][-1]:
                    sentence_for_x[obj] = (sentence, program, slot_dict, obj, sentence_len)

    for sentence, slot_program, slot_dict, obj, _ in sentence_for_x.values():
        obj1_string = gen_filter_string(slot_dict['OBJ1'], 'x')
        r1 = slot_dict['R1']
        obj2_string = gen_filter_string(slot_dict['OBJ2'], 'y')
        r2 = slot_dict['R2']
        obj3_string = gen_filter_string(slot_dict['OBJ3'], 'z')
        program = f'point(Object, lambda x: {obj1_string} and {r1}(x, iota(Object, lambda y: {obj2_string} and {r2}(y, iota(Object, lambda z: {obj3_string})))))'
        yield sentence, program, slot_program, slot_dict, obj


def random_sample_and_post(scene):
    unique_filters = [f for f in g_all_filter_ops if check_filter_unique(scene, f)]

    cat = random.choice(range(4)) + 1
    func = globals()[f'ground_program{cat}']

    for i in range(4):
        sols = list(func(scene, unique_filters))
        if len(sols) == 0:
            continue
        sentence, program, slot_program, slot_dict, obj = random.choice(sols)
        sentence = sentence.replace('left', 'left of')
        sentence = sentence.replace('right', 'right of')
        sentence = sentence.replace('-', '')

        return sentence, program, slot_program, slot_dict, obj

    print('Really bad...', scene['image_filename'])


def main():
    scenes = jacinle.load_json(args.scenes_json)['scenes']

    refexps = list()
    for scene_index, scene in enumerate(jacinle.tqdm(scenes[:150])):
        rv = random_sample_and_post(scene)
        if rv is None:
            continue
        sentence, program, slot_program, slot_dict, obj = rv
        refexps.append({
            'scene_index': scene_index,
            'image_filename': scene['image_filename'],
            'question': sentence,
            'program': program,
            'slot_program': program,
            'slot_dict': slot_dict,
            'answer': obj,
        })

    jacinle.dump_json(args.output, {'refexps': refexps[:100]})
    print('Saved: "{}".'.format(args.output))


if __name__ == '__main__':
    main()

