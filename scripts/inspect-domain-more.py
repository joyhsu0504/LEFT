#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect-domain-more.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/05/2023
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import csv
import traceback
from dataclasses import dataclass, field

import jacinle
import jacinle.io as io
from jaclearn.visualize.html_table import HTMLTableVisualizer, HTMLTableColumnDesc

from concepts.dsl.expression import VariableExpression, FunctionApplicationExpression, iter_exprs
from left.domain import create_bare_domain, create_default_parser

parser = jacinle.JacArgumentParser()
parser.add_argument('parsed_filename', type=str)
parser.add_argument('--output-dir', required=True, type=str, default='the output directory for the generated files')
args = parser.parse_args()


@dataclass
class FunctionGroupSummary(object):
    signature: str
    count: int = 0
    examples: dict = field(default_factory=dict)


def main():
    domain = create_bare_domain()
    parser = create_default_parser(domain)
    all_codes = io.load_pkl(args.parsed_filename)

    all_rows = list()
    all_function_groups: dict[str, FunctionGroupSummary] = dict()
    all_types: dict[str, list] = dict()
    for prompt, codes in jacinle.tqdm_gofor(all_codes, desc='Creating domain from parsings'):
        if isinstance(codes, str):
            codes = [codes]
        all_codes[prompt] = codes

        for code in codes:
            exception = ''
            parsed_expression = None
            parsed_expression_str = ''
            try:
                parsed_expression = parser.parse_expression(code)
                parsed_expression_str = str(parsed_expression)
            except:  # noqa: E722
                exception = traceback.format_exc()

            all_rows.append({
                'prompt': prompt,
                'raw_code': code,
                'parse_success': parsed_expression is not None,
                'parsed_expression': parsed_expression_str if parsed_expression is not None else exception,
            })

            if parsed_expression is not None:
                for expr in iter_exprs(parsed_expression):
                    if isinstance(expr, FunctionApplicationExpression):
                        function = expr.function
                        signature = get_function_signature(function)
                        if signature not in all_function_groups:
                            all_function_groups[signature] = FunctionGroupSummary(signature)
                        if function.name not in all_function_groups[signature].examples:
                            all_function_groups[signature].examples[function.name] = list()

                        all_function_groups[signature].count += 1
                        if len(all_function_groups[signature].examples[function.name]) < 3:
                            all_function_groups[signature].examples[function.name].append({
                                'prompt': prompt,
                                'raw_code': code,
                                'parsed_expression': '<pre>' + parsed_expression_str.replace(function.name + '(', f'<span style="color:red">{function.name}</span>(') + '</pre>',
                            })
                    elif isinstance(expr, VariableExpression):
                        typename = expr.return_type.typename
                        if typename not in all_types:
                            all_types[typename] = list()
                        if len(all_types[typename]) < 5:
                            all_types[typename].append({
                                'prompt': prompt,
                                'raw_code': code,
                                'parsed_expression': '<pre>' + parsed_expression_str.replace(typename, f'<span style="color:red">{typename}</span>') + '</pre>',
                            })

    io.mkdir(args.output_dir)

    with open(f'{args.output_dir}/summary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['# of prompts:', len(all_codes)])
        writer.writerow(['# of codes:', sum(len(x) for x in all_codes.values())])
        writer.writerow(['# of parsed codes:', sum(1 for x in all_rows if x['parse_success'])])
        writer.writerow(['# of parsed types:', len(domain.types)])
        writer.writerow(['# of parsed functions:', len(domain.functions)])
        writer.writerow(['# of parsed function groups:', len(all_function_groups)])
        writer.writerow([])
        for function_group in sorted(all_function_groups.values(), key=lambda x: x.count, reverse=True):
            writer.writerow([f'{function_group.signature}:', function_group.count])
    print('Summary written to', f'{args.output_dir}/summary.csv')

    visualizer = HTMLTableVisualizer(f'{args.output_dir}/parsing.html', 'Parsing Results')
    with visualizer.html():
        with visualizer.table('Result Summary', [
            HTMLTableColumnDesc('summary', 'Summary', 'code'),
        ]):
            string = ''
            string += f'# of prompts: {len(all_codes)}\n'
            string += f'# of codes: {sum(len(x) for x in all_codes.values())}\n'
            string += f'# of parsed codes: {sum(1 for x in all_rows if x["parse_success"])}\n'
            string += f'# of parsed types: {len(domain.types)}\n'
            string += f'# of parsed functions: {len(domain.functions)}\n'
            string += f'# of parsed function groups: {len(all_function_groups)}\n'
            visualizer.row(summary=string)
        with visualizer.table('Parsing Results', [
            HTMLTableColumnDesc('index', 'Index', 'text'),
            HTMLTableColumnDesc('prompt', 'Prompt', 'text', {}, {'width': '20%'}),
            HTMLTableColumnDesc('raw_code', 'Raw Code', 'code', {}, {'width': '20%'}),
            HTMLTableColumnDesc('parse_success', 'Parse Success', 'code', {'width': '50px'}),
            HTMLTableColumnDesc('parsed_expression', 'Parsed Expression', 'code', {}, {'width': '50%'}),
        ]):
            for i, row in enumerate(all_rows):
                visualizer.row(**row, index=i)
    print('Parsing results written to', f'{args.output_dir}/parsing.html')

    with open(f'{args.output_dir}/parsing.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'raw_code', 'parse_success', 'parsed_expression'])
        for row in all_rows:
            writer.writerow([row['prompt'], row['raw_code'], row['parse_success'], row['parsed_expression']])
    print('Parsing results written to', f'{args.output_dir}/parsing.csv')

    visualizer = HTMLTableVisualizer(f'{args.output_dir}/function_groups.html', 'Function Groups')
    with visualizer.html():
        for function_group in sorted(all_function_groups.values(), key=lambda x: x.count, reverse=True):
            with visualizer.table(f'{function_group.signature} (count = {function_group.count})', [
                HTMLTableColumnDesc('prompt', 'Prompt', 'text', None, {'width': '20%'}),
                HTMLTableColumnDesc('raw_code', 'Raw Code', 'code', None, {'width': '30%'}),
                HTMLTableColumnDesc('parsed_expression', 'Parsed Expression', 'raw', None, {'width': '50%'}),
            ]):
                for example_list in function_group.examples.values():
                    for example in example_list:
                        visualizer.row(**example)
    print('Function groups written to', f'{args.output_dir}/function_groups.html')

    visualizer = HTMLTableVisualizer(f'{args.output_dir}/types.html', 'Types')
    with visualizer.html():
        for typename, examples in all_types.items():
            with visualizer.table(f'{typename}', [
                HTMLTableColumnDesc('prompt', 'Prompt', 'text', None, {'width': '20%'}),
                HTMLTableColumnDesc('raw_code', 'Raw Code', 'code', None, {'width': '30%'}),
                HTMLTableColumnDesc('parsed_expression', 'Parsed Expression', 'raw', None, {'width': '50%'}),
            ]):
                for example in examples:
                    visualizer.row(**example)
    print('Types written to', f'{args.output_dir}/types.html')


def get_function_signature(function):
    argument_types = tuple(x.typename for x in function.ftype.argument_types)
    return_type = function.ftype.return_type.typename
    return f'{argument_types} -> {return_type}'


if __name__ == '__main__':
    main()

