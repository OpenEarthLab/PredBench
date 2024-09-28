import torch
import torch.nn as nn
from typing import Union, Tuple, Any
from mmengine.utils import is_tuple_of
from mmengine.analysis.print_helper import get_model_complexity_info as MMEngine_complexity_table

def measure_throughput(model, input_shape):
    def get_batch_size(one_input_shape):
        T, C, H, W = one_input_shape
        max_side = max(H, W)
        if max_side >= 128:
            bs = 5
            repetitions = 2000
        else:
            bs = 50
            repetitions = 200
        return bs, repetitions
    device = next(model.parameters()).device
    if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
        # input_shape = (t c h w)
        T, C, H, W = input_shape
        bs, repetitions = get_batch_size(input_shape)
        inputs = (torch.randn(bs, *input_shape).to(device), )
    elif is_tuple_of(input_shape, tuple) and all([
            is_tuple_of(one_input_shape, int)
            for one_input_shape in input_shape  # type: ignore
    ]):  # tuple of tuple of int, construct multiple tensors
        bs, repetitions = get_batch_size(input_shape[0])
        inputs = tuple([
            torch.randn(bs, *one_input_shape).to(device)
            for one_input_shape in input_shape  # type: ignore
        ])
    else:
        raise ValueError(
            '"input_shape" should be either a `tuple of int` (to construct'
            'one input tensor) or a `tuple of tuple of int` (to construct'
            'multiple input tensors).')
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(inputs, tuple):
                _ = model(*inputs)
            else:
                _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def get_model_computational_metrics(model, input_shape):
    complexity_results = MMEngine_complexity_table(model, input_shape)
    fps = measure_throughput(model, input_shape)
    return complexity_results['out_table'], fps