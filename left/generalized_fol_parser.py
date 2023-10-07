#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : generalized_fol_parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/07/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

import ast
from typing import Optional, List, Dict

from concepts.dsl.dsl_types import Variable, ObjectType, BOOL, INT64
from concepts.dsl.dsl_functions import Function, FunctionType, FunctionTyping
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.expression import GeneralizedQuantificationExpression, ValueOutputExpression, FunctionApplicationExpression, AndExpression, OrExpression, get_expression_definition_context, get_types
from concepts.dsl.parsers.fol_python_parser import FOLPythonParser


class NCGeneralizedFOLPythonParser(FOLPythonParser):
    def _is_quantification_expression_name(self, name: str) -> bool:
        return name in ['exists', 'forall', 'all', 'iota', 'describe', 'execute', 'point', 'count', 'view']

    def _parse_quantification_expression_inner(self, function_name: str, var: Variable, body: ast.Call, counting_quantifier: Optional[int] = None) -> ValueOutputExpression:
        ctx = get_expression_definition_context()
        if function_name in ['exists', 'forall']:
            assert var.dtype.typename in ['Object', 'Action'], f'Quantification variable must be of type Object or Action, got {var.dtype}.'
            rv = super()._parse_quantification_expression_inner(function_name, var, body)
            if rv.expression.return_type != BOOL:
                raise ValueError(f'Quantification expression must return a boolean, got {rv.expression.return_type}.')
            return rv
        elif function_name in ['all', 'iota']:
            if counting_quantifier is not None:
                function_name = (function_name, counting_quantifier)
            assert var.dtype.typename in ['Object', 'Action'], f'Quantification variable must be of type Object or Action, got {var.dtype}.'

            if var.dtype.typename == 'Object':
                if function_name == 'iota' and counting_quantifier is None:
                    # Conventional single-object iota.
                    return_type = self.domain.types['Object']
                else:
                    return_type = self.domain.types['ObjectSet']
            elif var.dtype.typename == 'Action':
                if function_name == 'iota' and counting_quantifier is None:
                    return_type = self.domain.types['Action']
                else:
                    raise NotImplementedError('Does not support ActionSet')
            else:
                raise TypeError(f'Unknown type name: {var.dtype.typename}.')

            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=return_type
                )
        elif function_name == 'describe':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for describe().'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=var.dtype
                )
        elif function_name == 'count':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for count().'
            assert var.dtype.typename == 'Object', f'Counting variable must be of type Object, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=INT64
                )
        elif function_name == 'execute':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for execute().'
            assert var.dtype.typename == 'Action', f'Execute variable must be of type Action, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=BOOL
                )
        elif function_name == 'point':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for point().'
            assert var.dtype.typename == 'Object', f'Point variable must be of type Object, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=var.dtype
                )
        elif function_name == 'view':
            assert counting_quantifier is None, 'Counting quantifier cannot be specified for view().'
            assert var.dtype.typename == 'Object', f'View variable must be of type Object, got {var.dtype}.'
            with ctx.with_variables(var):
                body = self._parse_expression_inner(body)
                if body.return_type != BOOL:
                    raise ValueError(f'Quantification expression must return a boolean, got {body.return_type}.')
                return GeneralizedQuantificationExpression(
                    function_name, var, body,
                    return_type=var.dtype
                )
        else:
            raise ValueError(f'Unknown quantification expression name: {function_name}.')

    def _parse_function_application(self, function_name: str, expression: ast.Call):
        if function_name == 'query':  # bypass query.
            assert len(expression.args) == 1, f'query() takes exactly one argument, got {len(expression.args)}: {ast.dump(expression)}'
            return self._parse_expression_inner(expression.args[0])
        else:
            return self._parse_function_application_simple(function_name, expression)

    def _parse_function_application_simple(self, function_name: str, expression: ast.Call) -> ValueOutputExpression:
        ctx = get_expression_definition_context()

        parsed_args = [self._parse_expression_inner(arg) for arg in expression.args]
        function = None

        if function_name not in ctx.domain.functions:
            # NB(Jiayuan Mao @ 2023/05/10): added to handle parsing failures.
            if function_name == 'and_':
                return AndExpression(*parsed_args)
            elif function_name == 'or_':
                return OrExpression(*parsed_args)

            if self.inplace_definition or self.inplace_polymorphic_function:
                assert self.inplace_polymorphic_function

                for arg in parsed_args:
                    if not isinstance(arg.return_type, ObjectType):
                        raise ValueError(f'In-place polymorphic function definition requires all arguments to be object-typed, got {arg.return_type}.')

                if self.inplace_polymorphic_function:
                    function_name = function_name + '_' + '_'.join([arg.return_type.typename for arg in parsed_args])

                if function_name in ctx.domain.functions:
                    function = ctx.domain.functions[function_name]
                else:
                    if self.inplace_definition:
                        function = Function(function_name, FunctionType(get_types(parsed_args), BOOL))
                        ctx.domain.define_function(function)
                    else:
                        raise KeyError(f'Function {function_name} is not defined in the domain.')
            else:
                raise KeyError(f'Function {function_name} is not defined in the domain.')
        else:
            function = ctx.domain.functions[function_name]
        return FunctionApplicationExpression(function, parsed_args)


if __name__ == '__main__':
    import re
    import csv
    import jacinle.io as io

    io.set_fs_verbose()

    def csv_summarize(codes: Dict[str, List[str]], csv_filename: str) -> FunctionDomain:
        domain = FunctionDomain()
        domain.define_type(ObjectType('Object'))
        domain.define_type(ObjectType('ObjectSet'))
        domain.define_type(ObjectType('Action'))

        # define built-in functions.
        domain.define_function(Function('equal', FunctionTyping[BOOL](INT64, INT64)))
        domain.define_function(Function('greater_than', FunctionTyping[BOOL](INT64, INT64)))
        domain.define_function(Function('less_than', FunctionTyping[BOOL](INT64, INT64)))

        parser = NCGeneralizedFOLPythonParser(domain, inplace_definition=True, inplace_polymorphic_function=True, inplace_definition_type=True)

        with open(csv_filename, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['prompt', 'code', 'success', 'parsing'])
            for prompt, codes in codes.items():
                for code in codes:
                    try:
                        parsing = parser.parse_expression(code)
                        csv_writer.writerow([prompt, code, True, str(parsing)])
                    except Exception as e:
                        csv_writer.writerow([prompt, code, False, str(e)])
        print(f'Parsing results are saved to {csv_filename}.')
        return domain

    with open('./codex/prompts/v3.txt') as f:
        prompts = f.read()

    results = re.findall(r'<text>(.*?)</text>\s*<code>([\s\S]*?)</code>', prompts)
    a = {x[0]: [x[1]] for x in results}
    csv_filename = './codex/results/prompts_v3.parsing.csv'
    csv_summarize(a, csv_filename)

    b = io.load_pkl('codex/results/clevr_small_v2.p')
    csv_filename = './codex/results/clevr_small_v2.parsing.csv'
    domain = csv_summarize(b, csv_filename)

    domain_filename = './codex/results/clevr_small_v2.domain.txt'
    with open(domain_filename, 'w') as f:
        f.write(domain.format_summary())
    print(f'Domain summary is saved to {domain_filename}.')

