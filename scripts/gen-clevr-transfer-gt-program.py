#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen-clevr-transfer-gt-program.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/14/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.


import jacinle

parser = jacinle.JacArgumentParser()
parser.add_argument('--files', nargs='+')
parser.add_argument('--output')

args = parser.parse_args()


parses = dict()

for file in args.files:
    queries = jacinle.load_json(file)
    list_of_queries = list(queries.values())[0]
    for q in list_of_queries:
        parses.setdefault(q['question'], []).append(q['program'])

jacinle.dump_json(args.output, parses)
print('Saved', args.output)
