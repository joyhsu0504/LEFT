#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-clevr-gt-program.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

from dataclasses import dataclass

import jacinle


def main(args):
    questions = jacinle.load(args.input)['questions']

    output = dict()
    for q in questions:
        question_str = q['question']
        program = q['program']
        fol_program_str = transform(program)

        output[question_str] = fol_program_str

        print(question_str)
        print(fol_program_str)

    jacinle.dump(args.output, output)


@dataclass
class QueryXProgram(object):
    full_program: str
    object_program: str


def get_op_type(op):
    if 'type' in op:
        return op['type']
    return op['function']


def transform(program):
    index_to_result = dict()
    variable_counter = 0

    for i, op in enumerate(program):
        op_type = get_op_type(op)
        if op_type == 'scene':
            variable_counter += 1
            index_to_result[i] = ('', f'x{variable_counter}')
        elif op_type in ('filter_size', 'filter_color', 'filter_material', 'filter_shape'):
            program_str, variable = index_to_result[op['inputs'][0]]
            this_program_str = f'{op["value_inputs"][0]}({variable})'
            program_str = this_program_str + ' and ' + program_str if program_str else this_program_str
            index_to_result[i] = (program_str, variable)
        elif op_type == 'unique':
            inner, variable = index_to_result[op['inputs'][0]]
            program_str = f'iota(Object, lambda {variable}: {inner})'
            index_to_result[i] = (program_str, None)
        elif op_type == 'relate':
            variable_counter += 1
            variable = f'x{variable_counter}'
            inner, _ = index_to_result[op['inputs'][0]]
            program_str = f'{op["value_inputs"][0]}({variable}, {inner})'
            index_to_result[i] = (program_str, variable)
        elif op_type in ('same_size', 'same_color', 'same_material', 'same_shape'):
            variable_counter += 1
            variable = f'x{variable_counter}'
            inner, _ = index_to_result[op['inputs'][0]]
            program_str = f'{op_type}({variable}, {inner})'
            index_to_result[i] = (program_str, variable)
        elif op_type == 'intersect' or op_type == 'union':
            e1, v1 = index_to_result[op['inputs'][1]]
            e2, v2 = index_to_result[op['inputs'][0]]

            if e1 == '':
                index_to_result[i] = (e2, v2)
            elif e2 == '':
                index_to_result[i] = (e1, v1)
            else:
                assert v1 in e1 and v2 in e2
                variable_counter += 1
                variable = f'x{variable_counter}'
                if op_type == 'intersect':
                    program_str = f'{e1.replace(v1, variable)} and {e2.replace(v2, variable)}'
                else:
                    program_str = f'({e1.replace(v1, variable)} or {e2.replace(v2, variable)})'
                index_to_result[i] = (program_str, variable)
        elif op_type in ('count', 'exist'):
            inner, variable = index_to_result[op['inputs'][0]]
            if inner == '':
                inner = f'thing({variable})'
            if op_type == 'exist':
                op_type = 'exists'
            program_str = f'{op_type}(Object, lambda {variable}: {inner})'
            index_to_result[i] = program_str
        elif op_type in ('query_shape', 'query_color', 'query_material', 'query_size'):
            metaconcept = op_type.split('_')[1]
            object_str, _ = index_to_result[op['inputs'][0]]
            program_str = f'describe({metaconcept.capitalize()}, lambda k: {metaconcept}(k, {object_str}))'
            index_to_result[i] = QueryXProgram(full_program=program_str, object_program=object_str)
        elif op_type == 'equal_integer':
            e1 = index_to_result[op['inputs'][0]]
            e2 = index_to_result[op['inputs'][1]]
            program_str = f'equal({e1}, {e2})'
            index_to_result[i] = program_str
        elif op_type in ('greater_than', 'less_than'):
            e1 = index_to_result[op['inputs'][0]]
            e2 = index_to_result[op['inputs'][1]]
            program_str = f'{op_type}({e1}, {e2})'
            index_to_result[i] = program_str
        elif op_type in ('equal_color', 'equal_material', 'equal_shape', 'equal_size'):
            e1 = index_to_result[op['inputs'][0]]
            e2 = index_to_result[op['inputs'][1]]
            op_type = op_type.replace('equal_', 'same_')
            program_str = f'{op_type}({e1.object_program}, {e2.object_program})'
            index_to_result[i] = program_str
        else:
            raise ValueError(f'Unknown op type: {op_type}, {op}')

    ret = index_to_result[len(program) - 1]
    if isinstance(ret, QueryXProgram):
        ret = ret.full_program
    assert isinstance(ret, str)
    return ret


if __name__ == '__main__':
    parser = jacinle.JacArgumentParser(usage='Generate ground-truth programs for CLEVR. Command: python gen-clevr-gt-program.py --input questions.json --output questions-genfol.json')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    main(args)

