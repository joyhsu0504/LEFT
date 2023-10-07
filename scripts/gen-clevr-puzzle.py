#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-clevr-puzzle.py
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


def filter(scene, name, input_):
    if name == 'object':
        return input_
    attribute = g_concept2attribute[name]
    return {i for i in input_ if scene['objects'][i][attribute] == name}


def multi_filter(scene, names, input_=None):
    if input_ is None:
        input_ = range(len(scene['objects']))
    for name in names.split():
        input_ = filter(scene, name, input_)
    return input_


def relate(scene, name, input_):
    if len(input_) != 1:
        raise ValueError()

    input_ = list(input_)[0]
    return set(scene['relationships'][name][input_])


def execute(scene, slot_dict):
    objs_for_i = dict()
    for i in range(1, 4 + 1):
        objs_for_i[i] = multi_filter(scene, slot_dict[f'OBJ{i}'])

    for objs in itertools.product(objs_for_i[1], objs_for_i[2], objs_for_i[3], objs_for_i[4]):
        succ = True
        for rel_i in range(5):
            if f'R{rel_i}' not in slot_dict:
                continue
            x, y, relation = slot_dict[f'R{rel_i}']
            x = objs[x - 1]
            y = objs[y - 1]
            if x not in scene['relationships'][relation][y]:
                succ = False
                break
        if succ:
            yield {1: objs[0], 2: objs[1], 3: objs[2], 4: objs[3]}


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
        f'{method}({vname})' for method in f.split() if method != 'object'
    ])


def get_possible_relations(scene, x, y):
    return [r for r in g_all_relate_ops if x in scene['relationships'][r][y]]


def gen(scene, nr_objects, nr_relations, make_wrong=False):
    if len(scene['objects']) < 8:
        return None

    object_to_nonunique = defaultdict(list)

    for f in g_all_filter_ops:
        objects = multi_filter(scene, f)
        if len(objects) > 1:
            for obj in objects:
                object_to_nonunique[obj].append(f)

    solution = None
    for trial in range(1000):
        object_indices = random.sample(range(len(scene['objects'])), nr_objects)
        slot_dict = dict()
        for i in range(1, nr_objects + 1):
            slot_dict[f'OBJ{i}'] = random.choice(object_to_nonunique[object_indices[i - 1]])
        relation_indices = random.sample(list(itertools.combinations(range(nr_objects), 2)), nr_relations)
        for i, (x, y) in enumerate(relation_indices):
            possible_relations = get_possible_relations(scene, object_indices[x], object_indices[y])
            if not possible_relations:
                break
            slot_dict[f'R{i}'] = (x + 1, y + 1, random.choice(possible_relations))

        solutions = list(execute(scene, slot_dict))
        if len(solutions) == 1:
            solution = slot_dict, solutions[0], object_indices
            break

    # print('Internal', solution)

    if solution is None:
        return None

    if make_wrong:
        slot_dict, _, _ = solution
        for trial in range(1000):
            rel_index = random.choice(range(nr_relations))
            new_slot_dict = slot_dict.copy()
            new_slot_dict[f'R{rel_index}'] = slot_dict[f'R{rel_index}'][:2] + (random.choice(g_all_relate_ops), )
            solutions = list(execute(scene, new_slot_dict))
            # print(new_slot_dict, solutions)
            if len(solutions) == 0:
                solution = new_slot_dict, None, None
                break

    if solution is None:
        return None

    return solution


def gen_sentence_and_program(slot_dict):
    fmt = 'Can you find four objects from the image such that: '

    constraints = list()
    program_parts = list()
    for i in range(1, 4 + 1):
        d = slot_dict[f'OBJ{i}']
        if d[0] in 'aeoiu':
            constraints.append(f'object {i} is an {d}')
        else:
            constraints.append(f'object {i} is a {d}')
        program_d = gen_filter_string(d, f'x{i}')
        if program_d != '':
            program_parts.append(program_d)
    for i in range(5):
        if f'R{i}' in slot_dict:
            x, y, relation = slot_dict[f'R{i}']
            if relation in ['left', 'right']:
                constraints.append(f'object {x} is {relation} of object {y}')
            else:
                constraints.append(f'object {x} is {relation} object {y}')
        program_parts.append(f'{relation}(x{x}, x{y})')

    return fmt + '; '.join(constraints) + '.', f'exists(Object, lambda x1: exists(Object, lambda x2: exists(Object, lambda x3: exists(Object, lambda x4: {" and ".join(program_parts)} ))))'


def main():
    scenes = jacinle.load_json(args.scenes_json)['scenes']

    # scene = scenes[0]
    # sol = gen(scene, 4, 3, make_wrong=True)
    # if sol is not None:
    #     slot_dict, solution, solution_gt = sol
    #     sentence, program = gen_sentence_and_program(slot_dict)

    #     print({
    #         'slot_dict': slot_dict,
    #         'solution': solution,
    #         'sentence': sentence,
    #         'program': program
    #     })

    puzzles = list()
    for scene_index, scene in enumerate(jacinle.tqdm(scenes)):
        if len(puzzles) == 100:
            break

        wrong = bool(random.choice(range(2)))
        desired_answer = not wrong
        sol = gen(scene, 4, 3, make_wrong=wrong)
        if sol is not None:
            slot_dict, solution, solution_gt = sol
            sentence, program = gen_sentence_and_program(slot_dict)
            puzzles.append({
                'image_index': scene_index,
                'image_filename': scene['image_filename'],
                'slot_dict': slot_dict,
                'solution': solution,
                'question': sentence,
                'program': program,
                'answer': desired_answer,
            })

    jacinle.dump_json(args.output, {'puzzles': puzzles[:100]})
    print('Saved: "{}".'.format(args.output))


if __name__ == '__main__':
    main()

