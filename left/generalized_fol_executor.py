#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 6-fol-customized-executor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/09/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F
import jacinle
import jactorch
import contextlib
from typing import Optional, Sequence, Tuple, List, Dict

import concepts.dsl.expression as E

from concepts.dsl.dsl_types import BOOL, INT64, Variable
from concepts.dsl.tensor_value import TensorValue
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.parsers.parser_base import ParserBase
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor

g_options = jacinle.FileOptions(__file__, use_self_mask=True, use_softmax_iota=True)


class ExecutionTraceGetter(object):
    def __init__(self, trace_obj):
        self.trace_obj = trace_obj

    def get(self) -> List[Tuple[E.Expression, TensorValue]]:
        return self.trace_obj


def _get_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return self_mask


def _do_apply_self_mask(m):
    if not g_options.use_self_mask:
        return m
    self_mask = _get_self_mask(m)
    return m * (1 - self_mask) + (-10) * self_mask


class NCGeneralizedFOLExecutor(FunctionDomainExecutor):
    def __init__(self, domain: FunctionDomain, parser: Optional[ParserBase] = None, allow_shift_grounding=False):
        super().__init__(domain, parser)
        # when allow_shift_grounding=true, grounding tensor of relations is a 1D vector with values corresponding to each entity, shifted through a linear layer.
        self.allow_shift_grounding = allow_shift_grounding
        self.variable_stack = dict()
        self.view_stack = list()
        self._record_execution_trace = False
        self._execution_trace = list()

    variable_stack: Dict[str, Variable]
    """A variable stack, used to store the variables that are used in the current scope."""
    view_stack: List[TensorValue]
    """A view stack, used to store the variables that are used for viewpoint anchoring."""

    _count_margin = 0.25
    _count_tau = 0.25

    @property
    def training(self):
        return self.grounding.training

    def _count(self, x: TensorValue) -> TensorValue:
        if self.training:
            return torch.sigmoid(x.tensor).sum(dim=-1)
        else:
            return (x.tensor > 0).sum(dim=-1).float()

    def greater_than(self, x: TensorValue, y: TensorValue) -> TensorValue:
        if self.training:
            rv = ((x.tensor - y.tensor - 1 + 2 * self._count_margin) / self._count_tau)
        else:
            rv = -10 + 20 * (x.tensor > y.tensor).float()
        return TensorValue(BOOL, [], rv, quantized=False)

    def less_than(self, x: TensorValue, y: TensorValue) -> TensorValue:
        return self.greater_than(y, x)

    def equal(self, x: TensorValue, y: TensorValue) -> TensorValue:
        if self.training:
            rv = ((2 * self._count_margin - (x.tensor - y.tensor).abs()) / (2 * self._count_margin) / self._count_tau)
        else:
            rv = -10 + 20 * (x.tensor == y.tensor).float()
        return TensorValue(BOOL, [], rv, quantized=False)

    @contextlib.contextmanager
    def record_execution_trace(self):
        self._record_execution_trace = True
        self._execution_trace = list()
        yield ExecutionTraceGetter(self._execution_trace)
        self._record_execution_trace = False
        self._execution_trace = None

    # @jacinle.log_function(verbose=True)
    def _execute(self, expr: E.Expression) -> TensorValue:
        rv = self._execute_inner(expr)

        if self._record_execution_trace:
            self._execution_trace.append((expr, rv))
        return rv

    def _execute_inner(self, expr: E.Expression) -> TensorValue:
        if isinstance(expr, E.BoolExpression):
            # NB (Jiayuan Mao @ 03/09): remove this test to handle "view & point" case.
            # for arg in expr.arguments:
            #     assert arg.return_type == BOOL
            # e.g., and, or, not
            if expr.bool_op is E.BoolOpType.AND:
                if isinstance(expr.arguments[0], E.GeneralizedQuantificationExpression) and expr.arguments[0].quantification_op == 'view':
                    assert len(expr.arguments) == 2
                    obj_anchor = self._execute(expr.arguments[0])
                    self.view_stack.append(obj_anchor)
                    try:
                        return self._execute(expr.arguments[1])
                    finally:
                        self.view_stack.pop()

                args = [self._execute(arg) for arg in expr.arguments]
                expanded_args = expand_argument_values(args)
                expanded_tensors = [a.tensor for a in expanded_args]
                result = torch.stack(expanded_tensors, dim=-1).amin(dim=-1)

                return TensorValue(expanded_args[0].dtype, expanded_args[0].batch_variables, result, quantized=False)
            elif expr.bool_op is E.BoolOpType.OR:
                args = [self._execute(arg) for arg in expr.arguments]
                expanded_args = expand_argument_values(args)
                expanded_tensors = [a.tensor for a in expanded_args]
                result = torch.stack(expanded_tensors, dim=-1).amax(dim=-1)
                return TensorValue(expanded_args[0].dtype, expanded_args[0].batch_variables, result, quantized=False)
            elif expr.bool_op is E.BoolOpType.NOT:
                args = [self._execute(arg) for arg in expr.arguments]
                assert len(args) == 1
                result = args[0].tensor
                result = torch.zeros_like(result) - result

                return TensorValue(args[0].dtype, args[0].batch_variables, result, quantized=False)
        elif isinstance(expr, E.FunctionApplicationExpression):
            if expr.function.name in self.function_implementations:
                # e.g., greater_than, equal, etc
                func = self.function_implementations[expr.function.name]
                args = [self._execute(arg) for arg in expr.arguments]
                return func(*args)

            else:
                # e.g., red, left, etc
                args = [self._execute(arg) for arg in expr.arguments]
                if len(args) == 1:
                    grounding_tensor = self.grounding.compute_similarity('attribute', expr.function.name)
                elif len(args) == 2:
                    if len(self.view_stack) > 0:
                        obj_anchor = self.view_stack[-1]
                        grounding_tensor = self.grounding.compute_similarity('multi_relation', expr.function.name)
                        grounding_tensor = torch.einsum('ijk,i->jk', grounding_tensor, obj_anchor.tensor)
                        # grounding_tensor = torch.einsum('ijk,k->ij', grounding_tensor, obj_anchor.tensor) # NS3D repro
                    else:
                        grounding_tensor = self.grounding.compute_similarity('relation', expr.function.name)
                        grounding_tensor = _do_apply_self_mask(grounding_tensor)
                        # grounding_tensor = grounding_tensor.T # NS3D repro
                else:
                    assert len(args) == 3
                    grounding_tensor = self.grounding.compute_similarity('multi_relation', expr.function.name)

                # e.g., temporal shift
                if self.allow_shift_grounding and len(args) == 2 and len(grounding_tensor.size()) == 1:
                    shift = True
                else:
                    shift = False

                batch_variable_names = list()
                dims_to_squeeze = list()
                for i, arg in enumerate(args):
                    if isinstance(arg, Variable):
                        batch_variable_names.append(arg.name)
                    else:
                        assert isinstance(arg, TensorValue)
                        if not shift:
                            grounding_tensor = (grounding_tensor * jactorch.add_dim_as_except(arg.tensor, grounding_tensor, i)).sum(i, keepdim=True)
                            dims_to_squeeze.append(i)

                for dim in reversed(dims_to_squeeze):
                    grounding_tensor = grounding_tensor.squeeze(dim)
                return TensorValue(BOOL, batch_variable_names, grounding_tensor, quantized=False)

        elif isinstance(expr, E.VariableExpression):
            # e.g., x, y
            assert expr.variable.name in self.variable_stack
            return self.variable_stack[expr.variable.name]

        elif isinstance(expr, E.ConstantExpression):
            # e.g., true, false, str, etc
            return expr.value

        elif isinstance(expr, E.QuantificationExpression):
            # e.g., for all, exists
            assert expr.variable.name not in self.variable_stack
            self.variable_stack[expr.variable.name] = expr.variable

            try:
                value = self._execute(expr.expression)
                variable_index = value.batch_variables.index(expr.variable.name)
                if expr.quantification_op is E.QuantificationOpType.FORALL:
                    return TensorValue(
                        value.dtype,
                        value.batch_variables[:variable_index] + value.batch_variables[variable_index + 1:],
                        value.tensor.amin(variable_index),
                        quantized=False
                    )
                elif expr.quantification_op is E.QuantificationOpType.EXISTS:
                    return TensorValue(
                        value.dtype,
                        value.batch_variables[:variable_index] + value.batch_variables[variable_index + 1:],
                        value.tensor.amax(variable_index),
                        quantized=False
                    )
                else:
                    raise ValueError(f'Unknown quantification op {expr.quantification_op}.')
            finally:
                del self.variable_stack[expr.variable.name]

        elif isinstance(expr, E.GeneralizedQuantificationExpression):
            # e.g., iota, point, describe, execute, count
            if expr.quantification_op == 'iota':
                assert expr.variable.name not in self.variable_stack
                self.variable_stack[expr.variable.name] = expr.variable

                try:
                    value = self._execute(expr.expression)
                    assert expr.variable.name in value.batch_variables, f'Variable {expr.variable.name} is not in {value.batch_variables}.'

                    if not g_options.use_softmax_iota:
                        return value

                    variable_index = value.batch_variables.index(expr.variable.name)
                    return TensorValue(
                        expr.return_type,
                        value.batch_variables,
                        F.softmax(value.tensor, dim=variable_index),
                        quantized=False
                    )
                finally:
                    del self.variable_stack[expr.variable.name]
            elif expr.quantification_op == 'point':
                assert expr.variable.name not in self.variable_stack
                self.variable_stack[expr.variable.name] = expr.variable

                try:
                    value = self._execute(expr.expression)
                    assert expr.variable.name in value.batch_variables, f'Variable {expr.variable.name} is not in {value.batch_variables}.'
                    variable_index = value.batch_variables.index(expr.variable.name)
                    return TensorValue(
                        expr.return_type,
                        value.batch_variables,
                        F.softmax(value.tensor, dim=variable_index),
                        quantized=False
                    )
                finally:
                    del self.variable_stack[expr.variable.name]
            elif expr.quantification_op == 'view':
                assert expr.variable.name not in self.variable_stack
                self.variable_stack[expr.variable.name] = expr.variable

                try:
                    value = self._execute(expr.expression)
                    assert expr.variable.name in value.batch_variables, f'Variable {expr.variable.name} is not in {value.batch_variables}.'
                    variable_index = value.batch_variables.index(expr.variable.name)
                    return TensorValue(
                        expr.return_type,
                        value.batch_variables,
                        F.softmax(value.tensor, dim=variable_index),
                        quantized=False
                    )
                finally:
                    del self.variable_stack[expr.variable.name]
            elif expr.quantification_op == 'describe':
                expr: E.GeneralizedQuantificationExpression
                assert isinstance(expr.expression, E.FunctionApplicationExpression)
                if expr.variable.dtype.typename == 'Object':
                    # e.g., what's the object? describe(Object, lambda x: object(x))
                    assert (
                        len(expr.expression.arguments) == 2 and
                        isinstance(expr.expression.arguments[0], E.VariableExpression) and expr.expression.arguments[0].variable.name == expr.variable.name and
                        expr.expression.arguments[1].return_type.typename in ['Object', 'Action']
                    )

                    value = self._execute(expr.expression.arguments[1])
                    assert len(value.batch_variables) == 1, f'Variable {expr.variable.name} is not the only batch variable in {value.batch_variables}.'
                    answer = self.grounding.compute_description('attribute', 'Shape')
                    # NB(Jiayuan Mao @ 2023/03/18): this does not support arbitrary arity!
                    answer = value.tensor @ answer
                    return TensorValue(expr.return_type, [], answer, quantized=False)
                elif expr.variable.dtype.typename == 'Action':
                    # e.g., what's the action? describe(Action, lambda x: action(x))
                    raise NotImplementedError('Describe not implemented for actions.')
                else:
                    # e.g., what's color of the object? describe(Color, lambda k: color(k, iota(Object, lambda x: wall(x)))
                    # e.g., what's direction of the action? describe(Direction, lambda k: direction(k, iota(Action, lambda x: run(x)))
                    # TODO(Jiayuan Mao @ 2023/03/18): generalize this to describe binary, ternary, etc. relationships.
                    assert (
                        len(expr.expression.arguments) == 2 and
                        isinstance(expr.expression.arguments[0], E.VariableExpression) and expr.expression.arguments[0].variable.name == expr.variable.name and
                        expr.expression.arguments[1].return_type.typename in ['Object', 'Action']
                    )

                    value = self._execute(expr.expression.arguments[1])
                    assert len(value.batch_variables) == 1, f'Variable {expr.variable.name} is not the only batch variable in {value.batch_variables}.'
                    answer = self.grounding.compute_description('attribute', expr.variable.dtype.typename)
                    # NB(Jiayuan Mao @ 2023/03/18): this does not support arbitrary arity!
                    answer = value.tensor @ answer
                    return TensorValue(expr.return_type, [], answer, quantized=False)
            elif expr.quantification_op == 'execute':
                assert isinstance(expr.expression, E.FunctionApplicationExpression) and len(expr.expression.arguments) == 3
                assert isinstance(expr.expression.arguments[0], E.VariableExpression) and expr.expression.arguments[0].variable.name == expr.variable.name

                object_1 = self._execute(expr.expression.arguments[1])
                object_2 = self._execute(expr.expression.arguments[2])
                return self.grounding.compute_action(object_1, object_2, expr.expression.function.name)
            elif expr.quantification_op == 'count':
                assert expr.variable.name not in self.variable_stack
                self.variable_stack[expr.variable.name] = expr.variable

                try:
                    value = self._execute(expr.expression)
                    assert expr.variable.name in value.batch_variables, f'Variable {expr.variable.name} is not in {value.batch_variables}.'
                    result = self._count(value)
                    return TensorValue(INT64, value.batch_variables, result, quantized=False)
                finally:
                    del self.variable_stack[expr.variable.name]

        else:
            raise ValueError(f'Unknown expression type {type(expr)}.')


def expand_argument_values(argument_values: Sequence[TensorValue]) -> List[TensorValue]:
    """Expand a list of argument values to the same batch size.
    Args:
        argument_values: a list of argument values.
    Returns:
        the result list of argument values. All return values will have the same batch size.
    """
    has_slot_var = False
    for arg in argument_values:
        if isinstance(arg, TensorValue):
            for var in arg.batch_variables:
                if var == '??':
                    has_slot_var = True
                    break
    if has_slot_var:
        return list(argument_values)

    if len(argument_values) < 2:
        return list(argument_values)

    argument_values = list(argument_values)
    batch_variables = list()
    batch_sizes = list()
    for arg in argument_values:
        if isinstance(arg, TensorValue):
            for var in arg.batch_variables:
                if var not in batch_variables:
                    batch_variables.append(var)
                    batch_sizes.append(arg.get_variable_size(var))
        else:
            assert isinstance(arg, (int, slice)), arg

    masks = list()
    for i, arg in enumerate(argument_values):
        if isinstance(arg, TensorValue):
            argument_values[i] = arg.expand(batch_variables, batch_sizes)
            if argument_values[i].tensor_mask is not None:
                masks.append(argument_values[i].tensor_mask)

    if len(masks) > 0:
        final_mask = torch.stack(masks, dim=-1).amin(dim=-1)
        for arg in argument_values:
            if isinstance(arg, TensorValue):
                arg.tensor_mask = final_mask
                arg._mask_certified_flag = True  # now we have corrected the mask.
    return argument_values
