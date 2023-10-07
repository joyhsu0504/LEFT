#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect-domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/18/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

import jacinle

from left.domain import make_domain

parser = jacinle.JacArgumentParser()
parser.add_argument('parsed_filename', type=str)
args = parser.parse_args()


def main():
    domain = make_domain(args.parsed_filename)
    domain.print_summary()

    print('Summary:')
    print('  - # of types: {}'.format(len(domain.types)))
    print('  - # of functions: {}'.format(len(domain.functions)))

    # Group functions by their arguments and return types.
    function_groups = dict()
    for function in domain.functions.values():
        argument_types = tuple(x.typename for x in function.ftype.argument_types)
        return_type = function.ftype.return_type.typename
        key = f'{argument_types} -> {return_type}'
        function_groups.setdefault(key, []).append(function)

    print('  - # of function groups: {}'.format(len(function_groups)))
    for key, functions in sorted(function_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print('    - {}: {}'.format(key, len(functions)))


if __name__ == '__main__':
    main()

