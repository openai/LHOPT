import abc
import collections
import dataclasses
import enum
import functools
import random
from typing import Any, Dict, Set, Tuple, Union

import numpy as np
import scipy.stats
import torch

from . import registry


# using an IntEnum because the levels are ordered
class Levels(enum.IntEnum):
    INNER_STEP = 0
    PARAM_GROUP = 1
    PARAM = 2


# ############################ utility functions ############################


def cached_lazy(func):
    cache = []

    @functools.wraps(func)
    def cached_lazy_inner():
        if not cache:
            cache.append(func())
        return cache[0]

    # add cache to metadata to be able to check if it's realized
    cached_lazy_inner.cache = cache
    return cached_lazy_inner


def value_to_lazy(v):
    cache = [v]

    def value_to_lazy_inner():
        return cache[0]

    # add cache to metadata to be able to check if it's realized
    value_to_lazy_inner.cache = cache
    return value_to_lazy_inner


# #################################### Op's ####################################


# just for typing
class BaseOp:
    pass


@dataclasses.dataclass(eq=True, frozen=True)
class InputOp(BaseOp):
    # string corresponding to input
    key: str
    # which level it is used in
    level: Levels


@dataclasses.dataclass(eq=True, frozen=True)
class AggregatorOp(BaseOp):
    # type of aggregation
    key: str
    # which level it is used in
    level: Levels
    # either:
    # - another Op from the same level
    # - an AggregatorOp from the lower level
    target: Union[BaseOp]
    # a tuple of key-value pairs (so that it is hashable)
    # keys being strings
    # values being hashable values
    kwargs: Tuple[Tuple[Any]] = ()


@dataclasses.dataclass(eq=True, frozen=True)
class FunctionOp(BaseOp):
    # type of function
    key: str
    # which level it is used in
    level: Levels
    # tuple of values
    # values being Op's or hashable values
    args: Tuple[Any] = ()
    # a tuple of key-value pairs (so that it is hashable)
    # keys being strings
    # values being Op's or hashable values
    kwargs: Tuple[Tuple[Any]] = ()


# ############################### Aggregator's ###############################

register_aggregator = registry.build_register_decorator("INNER_FEATURES_AGGREGATORS")
lookup_aggregator = registry.build_lookup("INNER_FEATURES_AGGREGATORS")


class Aggregator(abc.ABC):
    def __init__(self):
        self.reset()

    @abc.abstractmethod
    def append(self, lazy_value):
        pass

    @abc.abstractmethod
    def aggregate(self) -> Any:
        # returns a lazy value
        pass

    @abc.abstractmethod
    def reset(self):
        pass


@register_aggregator("concat")
class ConcatAggregator(Aggregator):
    def append(self, lazy_value):
        self.values.append(lazy_value())

    def aggregate(self):
        assert len(self.values) > 0
        return value_to_lazy(self.values)

    def reset(self):
        self.values = []


@register_aggregator("mean")
class MeanAggregator(Aggregator):
    def append(self, lazy_value):
        v = lazy_value()
        new_count = self.count + 1
        self.value = self.value * (self.count / new_count) + v / new_count
        self.count = new_count

    def aggregate(self):
        assert self.count > 0
        return value_to_lazy(self.value)

    def reset(self):
        self.count = 0
        self.value = 0.0


@register_aggregator("optional_mean")
class OptionalMeanAggregator(Aggregator):
    """
    similar to MeanAggregator, but ignores None values and returns
    None if we've not averaged any value
    """

    def append(self, lazy_value):
        v = lazy_value()
        if v is not None:
            new_count = self.count + 1
            self.value = self.value * (self.count / new_count) + v / new_count
            self.count = new_count

    def aggregate(self):
        if self.count > 0:
            value = self.value
        else:
            value = None
        return value_to_lazy(value)

    def reset(self):
        self.count = 0
        self.value = 0.0


@register_aggregator("stochastic_mean")
class StochasticMeanAggregator(OptionalMeanAggregator):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        super().__init__()

    def append(self, lazy_value):
        if random.random() < self.keep_prob:
            super().append(lazy_value)


@register_aggregator("stochastic_mean_with_last")
class StochasticMeanWithLastAggregator(OptionalMeanAggregator):
    """
    similar to StochasticMeanAggregator, but always includes the last element
    (to make sure we always have valid values)
    """

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        super().__init__()

    def append(self, lazy_value):
        self.last_value = lazy_value
        self.did_append_last = random.random() < self.keep_prob
        if self.did_append_last:
            super().append(lazy_value)

    def aggregate(self):
        assert self.last_value is not None
        if not self.did_append_last:
            super(StochasticMeanWithLastAggregator, self).append(self.last_value)
        assert self.count > 0
        return super().aggregate()

    def reset(self):
        # reset mean state
        super().reset()
        self.last_value = None
        self.did_append_last = None


@register_aggregator("lazy_mean")
class LazyMeanAggregator(Aggregator):
    def append(self, lazy_value):
        self.lazy_values.append(lazy_value)

    def aggregate(self):
        assert len(self.lazy_values) > 0
        # make a local reference to the current lazy values list
        lazy_values = self.lazy_values

        @cached_lazy
        def lazy_mean_inner():
            count = len(lazy_values)
            return sum(v() for v in lazy_values) / count

        return lazy_mean_inner

    def reset(self):
        self.lazy_values = []


@register_aggregator("last")
class LastAggregator(Aggregator):
    def append(self, lazy_value):
        self.last_value = lazy_value

    def aggregate(self):
        assert self.last_value is not None
        return self.last_value()

    def reset(self):
        self.last_value = None


@register_aggregator("first")
class FirstAggregator(Aggregator):
    def append(self, lazy_value):
        if self.current_value is None:
            self.current_value = lazy_value()

    def aggregate(self):
        return self.current_value

    def reset(self):
        self.current_value = None


# ######################### inner feature functions #########################

register_function = registry.build_register_decorator("INNER_FEATURES_FUNCTIONS")
lookup_function = registry.build_lookup("INNER_FEATURES_FUNCTIONS")


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.cpu().item()
    else:
        return v


@register_function("add")
def fn_add(a, b):
    return a() + b()


@register_function("sub")
def fn_sub(a, b):
    return a() - b()


@register_function("div")
def fn_div(num, denom):
    return num() / denom()


@register_function("gt")
def fn_gt(a, b):
    # greater than
    return a() > b()


def _norm(v):
    if isinstance(v, torch.Tensor):
        return _to_scalar(torch.norm(v))
    else:
        # assume numpy
        return np.linalg.norm(v)


@register_function("norm")
def fn_norm(t):
    v = t()
    return _norm(v)


@register_function("abs")
def fn_abs(t):
    return abs(t())


def _log(v):
    if isinstance(v, torch.Tensor):
        return torch.log(v)
    else:
        return np.log(v)


@register_function("log")
def fn_log(t):
    v = t()
    return _log(v)


@register_function("tensor_mean")
def fn_tensor_mean(t):
    v = t()
    if isinstance(v, torch.Tensor):
        # cast to float, in case v is boolean
        return v.to(torch.float).mean().cpu().item()
    else:
        return v.mean()


def _cosine_similarity(a, b, epsilon):
    dot_prod = _to_scalar(a.flatten().dot(b.flatten()))
    denom = max(_norm(a) * _norm(b), epsilon)
    return dot_prod / denom


@register_function("cosine_similarity")
def fn_cosine_similarity(a, b, epsilon):
    t1 = a()
    t2 = b()
    eps = epsilon()
    return _cosine_similarity(t1, t2, eps)


@register_function("scaled_cosine_similarity")
def fn_scaled_cosine_similarity(a, b, epsilon):
    t1 = a()
    t2 = b()
    eps = epsilon()
    cos = _cosine_similarity(t1, t2, eps)
    dim = np.prod(t1.shape)
    return np.sqrt(dim) * cos


@register_function("normal_cdf")
def fn_normal_cdf(t):
    return scipy.stats.norm.cdf(t())


def _clip(v, lower, upper):
    if isinstance(v, torch.Tensor):
        return torch.clamp(v, lower, upper)
    else:
        return np.clip(v, lower, upper)


@register_function("logit")
def fn_logit(t, epsilon):
    p = t()
    eps = epsilon()
    p = _clip(p, eps, 1 - eps)
    return _log(p / (1 - p))


# ######################### putting it all together #########################


class InnerStepFeatures(object):
    def __init__(self):
        # keep a list of output op's (to maintain order)
        self.output_ops = []

        self.op_by_level: Dict[Levels, Set[BaseOp]] = collections.defaultdict(set)
        self.aggregator_op_to_obj: Dict[AggregatorOp, Aggregator] = {}
        self.inputs_by_level: Dict[Levels, Dict[str, Any]] = {}
        for level in Levels:
            self.inputs_by_level[level] = {}

    def _register_op(self, op):
        # make registration idempotent
        if op not in self.op_by_level[op.level]:
            self.op_by_level[op.level].add(op)

            def register_maybe_sub_op(maybe_sub_op):
                # function for arguments that may be sub op's
                # if it is a sub op, then check for validity as well
                if isinstance(maybe_sub_op, BaseOp):
                    if isinstance(maybe_sub_op, AggregatorOp):
                        # sub aggregator op must be of a lower level
                        assert op.level == maybe_sub_op.level - 1
                    else:
                        # otherwise must be in the same level as the parent
                        assert op.level == maybe_sub_op.level
                    self._register_op(maybe_sub_op)

            if isinstance(op, InputOp):
                # don't need to do anything
                # note: we could optionally store the input ops to not save
                # unused inputs
                pass
            elif isinstance(op, AggregatorOp):
                register_maybe_sub_op(op.target)
                aggregator = lookup_aggregator(op.key)(**dict(op.kwargs))
                self.aggregator_op_to_obj[op] = aggregator
            elif isinstance(op, FunctionOp):
                # don't run the function, but test that it is there
                lookup_function(op.key)
                for arg in op.args:
                    register_maybe_sub_op(arg)
                for key, kwarg in op.kwargs:
                    register_maybe_sub_op(kwarg)
            else:
                raise NotImplementedError

    def register_output(self, op):
        self.output_ops.append(op)
        self._register_op(op)

    def _record_input(self, name, level, value, lazy):
        if not lazy:
            value = value_to_lazy(value)
        # assume it's always lazy
        self.inputs_by_level[level][name] = value

    def _aggregate(self, level):
        done: Set[BaseOp] = set()
        values: Dict[BaseOp, Any] = {}
        none_fn = value_to_lazy(None)

        def calculate_op(op):
            # use cache, if available
            if op not in values:
                # if op is in done but doesn't have a value,
                # we may have a cyclic loop
                assert op not in done

                if isinstance(op, InputOp):
                    return self.inputs_by_level[level].get(op.key, none_fn)
                elif isinstance(op, AggregatorOp):
                    # aggregator must be of a lower level
                    assert op.level == level + 1
                    aggregator = self.aggregator_op_to_obj[op]
                    value = aggregator.aggregate()
                    # after accessing state, we reset it here
                    # (this should be done exactly once)
                    aggregator.reset()
                elif isinstance(op, FunctionOp):
                    args = []
                    for arg in op.args:
                        if isinstance(arg, BaseOp):
                            # recurse down the compute graph
                            v = calculate_op(arg)
                            args.append(v)
                        else:
                            args.append(value_to_lazy(arg))

                    kwargs = {}
                    for key, kwarg in op.kwargs:
                        if isinstance(kwarg, BaseOp):
                            # recurse down the compute graph
                            v = calculate_op(kwarg)
                            kwargs[key] = v
                        else:
                            kwargs[key] = value_to_lazy(kwarg)

                    func = lookup_function(op.key)

                    @cached_lazy
                    def function_op_apply():
                        return func(*args, **kwargs)

                    # make function application lazy
                    # (so that aggregators can choose whether
                    # or not to realize the values)
                    value = function_op_apply
                else:
                    raise NotImplementedError

                values[op] = value

            return values[op]

        # update aggregators at this level
        for op in self.op_by_level[level]:
            if isinstance(op, AggregatorOp):
                self.aggregator_op_to_obj[op].append(calculate_op(op.target))

        # reset inputs for this level
        self.inputs_by_level[level] = {}

    def record_per_inner_step(self, name, value, lazy=False):
        self._record_input(name, Levels.INNER_STEP, value, lazy)

    def record_per_param_group(self, name, value, lazy=False):
        self._record_input(name, Levels.PARAM_GROUP, value, lazy)

    def record_per_param(self, name, value, lazy=False):
        self._record_input(name, Levels.PARAM, value, lazy)

    def _cached_lazy(self, func, level):
        inner = cached_lazy(func)
        self._record_input(name=func.__name__, level=level, value=inner, lazy=True)
        return inner

    def cached_lazy_per_inner_step(self, func):
        return self._cached_lazy(func, Levels.INNER_STEP)

    def cached_lazy_per_param_group(self, func):
        return self._cached_lazy(func, Levels.PARAM_GROUP)

    def cached_lazy_per_param(self, func):
        return self._cached_lazy(func, Levels.PARAM)

    def param_aggregate(self):
        self._aggregate(Levels.PARAM)

    def param_group_aggregate(self):
        self._aggregate(Levels.PARAM_GROUP)

    def inner_step_aggregate(self):
        self._aggregate(Levels.INNER_STEP)

    def outer_step_aggregate(self):
        # return values from all inner step aggregators
        self.outer_step_values = {}
        with torch.no_grad():
            for op in self.op_by_level[Levels.INNER_STEP]:
                if isinstance(op, AggregatorOp):
                    aggregator = self.aggregator_op_to_obj[op]
                    value = aggregator.aggregate()
                    # after accessing state, we reset it here
                    # (this should be done exactly once)
                    aggregator.reset()
                    self.outer_step_values[op] = value()
