#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : prune-domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import csv
import traceback
from dataclasses import dataclass, field

import jacinle
import jacinle.io as io
from jaclearn.visualize.html_table import HTMLTableVisualizer, HTMLTableColumnDesc

import concepts.dsl.expression as E
from concepts.dsl.dsl_types import BOOL, INT64
from concepts.dsl.expression import FunctionApplicationExpression, iter_exprs
from concepts.dsl.function_domain import FunctionDomain
from left.domain import create_bare_domain, create_default_parser

parser = jacinle.JacArgumentParser()
parser.add_argument('parsed_filename', type=str)
parser.add_argument('--output', type=str, help='output parsed expressions')
parser.add_argument('--debug-checker', action='store_true')
args = parser.parse_args()


@dataclass
class FunctionGroupSummary(object):
    signature: str
    count: int = 0
    examples: dict[str, list[dict]] = field(default_factory=dict)


def main():
    domain = create_bare_domain()
    parser = create_default_parser(domain)
    all_codes = io.load_pkl(args.parsed_filename)

    for prompts, codes in jacinle.tqdm_gofor(all_codes):
        for code in codes:
            try:
                _ = parser.parse_expression(code)
            except Exception:
                pass

    print('Summary (before pruning):')
    print('  - # of types: {}'.format(len(domain.types)))
    print('  - # of functions: {}'.format(len(domain.functions)))
    domain = prune_domain(domain)

    print('Summary (after pruning):')
    print('  - # of types: {}'.format(len(domain.types)))
    print('  - # of functions: {}'.format(len(domain.functions)))
    print('-' * 80)
    domain.print_summary()


def main2():
    domain = create_bare_domain()
    parser = create_default_parser(domain)

    if args.parsed_filename.endswith('.json'):
        all_codes = io.load_json(args.parsed_filename)
    else:
        all_codes = io.load_pkl(args.parsed_filename)

    expressions = list()
    for prompt, codes in jacinle.tqdm_gofor(all_codes, leave=False, desc='Parsing'):
        if isinstance(codes, str):
            codes = [codes]
        for code in codes:
            code = code.strip()
            try:
                expr = parser.parse_expression(code)
                expressions.append((prompt, code, expr))
            except Exception:
                # traceback.print_exc()
                # import ipdb; ipdb.set_trace()
                pass

    print('Summary (before pruning):')
    print('  - # of types: {}'.format(len(domain.types)))
    print('  - # of functions: {}'.format(len(domain.functions)))
    print('  - # of input sentences: {}'.format(len(all_codes)))
    print('  - # of input expressions: {}'.format(sum(len(c) for c in all_codes.values())))
    print('  - # of parsed sentences: {}'.format(len({x[0] for x in expressions})))
    print('  - # of parsed expressions: {}'.format(len(expressions)))

    checked_expressions = list()
    for prompt, code, expr in jacinle.tqdm(expressions, leave=False, desc='Checking expressions'):
        if args.debug_checker:
            print(expr)
        try:
            check_expr_validity(expr)
            if args.debug_checker:
                print('  - OK')
            checked_expressions.append((prompt, code, expr))
        except Exception:
            if args.debug_checker:
                print('  - failed')
                traceback.print_exc()
                input('Press any key to continue...')

    domain = create_bare_domain()
    parser = create_default_parser(domain)
    for _, code, _ in jacinle.tqdm(checked_expressions, leave=False, desc='Re-parsing'):
        try:
            parser.parse_expression(code)
        except Exception:
            pass

    print('Summary (after pruning):')
    print('  - # of types: {}'.format(len(domain.types)))
    print('  - # of functions: {}'.format(len(domain.functions)))
    print('  - # of parsed sentences: {}'.format(len({s[0] for s in checked_expressions})))
    print('  - # of parsed expressions: {}'.format(len(checked_expressions)))

    print('-' * 80)
    domain.print_summary()

    if args.output is not None:
        expressions = dict()
        for prmopt, code, expr in checked_expressions:
            if prmopt not in expressions:
                expressions[prmopt] = list()
            expressions[prmopt].append(code)
        io.dump(args.output, expressions)
        print(f'Output to {args.output}.')


def prune_domain(old_domain: FunctionDomain) -> FunctionDomain:
    new_domain = create_bare_domain()

    # from IPython import embed; embed()

    for name, function in old_domain.functions.items():
        if name in new_domain.functions:
            continue

        print('Checking function: {} {}'.format(name, function))

        ftype = function.ftype
        argument_types = [x.typename for x in ftype.argument_types]
        return_type = ftype.return_type.typename

        pass_test = False
        if len(argument_types) > 1 and all(x == 'Object' for x in argument_types) and return_type == 'bool':
            pass_test = True

        if len(argument_types) > 1 and all(x == 'Object' for x in argument_types[1:]) and return_type == 'bool':
            pass_test = True

        if pass_test:
            print(f'  Pass test: {name} {argument_types} -> {return_type}')
            new_domain.functions[name] = function
            for t in ftype.argument_types:
                if t.typename not in new_domain.types:
                    new_domain.types[t.typename] = t
        else:
            print(f'  Prune {name}')

    return new_domain


def check_expr_validity(expression: E.Expression):
    if isinstance(expression, E.GeneralizedQuantificationExpression):
        if expression.quantification_op in ('describe', 'count'):
            pass
        else:
            raise ValueError('Invalid quantification op: {}'.format(expression.quantification_op))

    def dfs(expr: E.Expression, allow_queries: bool = False):
        if isinstance(expr, E.GeneralizedQuantificationExpression):
            if expr.quantification_op == 'iota':
                dfs(expr.expression, allow_queries=allow_queries)
            elif expr.quantification_op == 'point':
                assert allow_queries
                dfs(expr.expression, allow_queries=False)
            elif expr.quantification_op == 'view':
                raise ValueError(f'Invalid view: {repr(expr)}.')
            elif expr.quantification_op == 'describe':
                assert allow_queries
                if isinstance(expr.expression, E.FunctionApplicationExpression):
                    if expr.variable.dtype.typename == 'Object':
                        pass
                    elif expr.variable.dtype.typename == 'Action':
                        pass
                    else:
                        if (
                            len(expr.expression.arguments) == 2 and
                            isinstance(expr.expression.arguments[0], E.VariableExpression) and expr.expression.arguments[0].variable.name == expr.variable.name and
                            expr.expression.arguments[1].return_type.typename in ['Object', 'Action']
                        ):
                            return dfs(expr.expression.arguments[1], allow_queries=allow_queries)
                        else:
                            raise ValueError(f'Invalid describe: {repr(expr)}.')
                else:
                    raise ValueError(f'Invalid describe: {repr(expr)}.')
                dfs(expr.expression, allow_queries=False)
            elif expr.quantification_op == 'count':
                dfs(expr.expression, allow_queries=allow_queries)
        elif isinstance(expr, FunctionApplicationExpression):
            if expr.return_type in (BOOL, ):
                function = expr.function
                if function.name in ('equal', 'less_than', 'greater_than'):
                    pass
                elif len(function.ftype.argument_types) > 0 and all(x.typename == 'Object' for x in function.ftype.argument_types):
                    # TODO(Jiayuan Mao @ 2023/04/11): add a flag control to this.
                    if len(function.ftype.arguments) in (1, 2):
                        pass
                    else:
                        raise ValueError(f'Invalid function: {repr(expr)}.')
                else:
                    raise ValueError('Invalid function: {}'.format(function))
            else:
                raise ValueError('Invalid return type: {}'.format(expr.return_type))
            for arg in expr.arguments:
                dfs(arg, allow_queries=allow_queries)
        elif isinstance(expr, E.VariableExpression):
            pass
        elif isinstance(expr, E.ConstantExpression):
            pass
        elif isinstance(expr, E.BoolExpression):
            for arg in expr.arguments:
                # NB(Jiayuan Mao @ 2023/04/11): technically should add view and point.
                if arg.return_type != BOOL:
                    raise ValueError('Invalid bool expression: {}'.format(arg))
            for arg in expr.arguments:
                dfs(arg, allow_queries=allow_queries)
        elif isinstance(expr, E.QuantificationExpression):
            dfs(expr.expression, allow_queries=allow_queries)
        else:
            raise ValueError('Invalid expression: {}'.format(repr(expr)))

    dfs(expression, allow_queries=True)


if __name__ == '__main__':
    main2()

