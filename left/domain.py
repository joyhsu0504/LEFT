#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : definition.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/22/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.


"""Domain creation from Codex/Llama parsing."""

import jacinle
import jacinle.io as io
from typing import Optional, Tuple, List, Dict

from concepts.dsl.dsl_types import ObjectType, BOOL, INT64
from concepts.dsl.dsl_functions import Function, FunctionTyping
from concepts.dsl.function_domain import FunctionDomain
from left.generalized_fol_parser import NCGeneralizedFOLPythonParser


__all__ = [
    'create_bare_domain', 'create_default_parser',
    'make_domain', 'read_concepts_v1', 'read_concepts_v2',
    'read_description_categories', 'create_domain_from_parsing'
]


def create_bare_domain() -> FunctionDomain:
    domain = FunctionDomain('Left')
    domain.define_type(ObjectType('Object'))
    domain.define_type(ObjectType('Object_Set'))
    domain.define_type(ObjectType('Action'))

    # define built-in functions.
    domain.define_function(Function('equal', FunctionTyping[BOOL](INT64, INT64)))
    domain.define_function(Function('greater_than', FunctionTyping[BOOL](INT64, INT64)))
    domain.define_function(Function('less_than', FunctionTyping[BOOL](INT64, INT64)))

    return domain


def create_default_parser(domain: FunctionDomain) -> NCGeneralizedFOLPythonParser:
    parser = NCGeneralizedFOLPythonParser(domain, inplace_definition=True, inplace_polymorphic_function=True, inplace_definition_type=True)
    return parser


def create_domain_from_parsing(codes: Dict[str, List[str]]) -> FunctionDomain:
    domain = create_bare_domain()
    parser = create_default_parser(domain)

    for prompt, codes in jacinle.tqdm_gofor(codes, desc='Creating domain from parsings'):
        if isinstance(codes, str):
            codes = [codes]

        for code in codes:
            try:
                _ = parser.parse_expression(code)
            except Exception as e:  # noqa: E722
                print(e)
                # NB(Jiayuan Mao @ 2023/04/05): basically ignores all parsing errors.
                continue

    return domain


def read_concepts_v1(domain: FunctionDomain) -> Tuple[List[str], List[str], List[str]]:
    ds_functions = list(domain.functions.keys())

    attribute_concepts, relational_concepts, multi_relational_concepts = [], [], []
    for f in ds_functions:
        if '_Object_Object_Object' in f:
            multi_relational_concepts.append(f)
        elif '_Object_Object' in f:
            relational_concepts.append(f)
        elif '_Object' in f:
            attribute_concepts.append(f)
        else:
            pass

    attribute_concepts.sort()
    relational_concepts.sort()
    multi_relational_concepts.sort()

    return attribute_concepts, relational_concepts, multi_relational_concepts


def get_arity(function: Function) -> Optional[int]:
    ftype = function.ftype
    if ftype.return_type != BOOL:
        return None

    for arg_type in ftype.argument_types:
        if arg_type.typename not in ['Object', 'Object_Set', 'Action']:
            return None

    return len(ftype.argument_types)


def read_concepts_v2(domain: FunctionDomain) -> Tuple[List[str], List[str], List[str]]:
    functions = {1: list(), 2: list(), 3: list()}
    for name, function in domain.functions.items():
        arity = get_arity(function)
        if arity is not None and 1 <= arity <= 3:
            functions[arity].append(name)

    return functions[1], functions[2], functions[3]


def read_description_categories(domain: FunctionDomain) -> Tuple[List[str]]:
    # TODO(Jiayuan Mao @ 2023/03/18): if we want to describe higher arity relations, we need to check the actual arity of the function.
    output = list()
    for name, t in domain.types.items():
        if t.typename not in ('Object', 'Object_Set', 'Action'):
            output.append(name)
    return output


def make_domain(parsed_test_path: str) -> FunctionDomain:
    codes = io.load_pkl(parsed_test_path)
    domain = create_domain_from_parsing(codes)
    return domain

