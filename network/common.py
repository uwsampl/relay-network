
from tvm.relay import op, var, Var, Function, Clause, PatternConstructor, PatternVar, Match, const
from tvm.relay import TupleGetItem, Tuple, TensorType, TupleType, If
from network import Network
import numpy as np

class Linear(Network):
    def build_impl(self, input_size, output_size, dtype="float32"):
        x = self.input(var("linear_input", shape=(1, input_size), dtype=dtype))
        w = self.weight(var("linear_weight", shape=(output_size, input_size), dtype=dtype))
        b = self.weight(var("linear_bias", shape=(output_size,), dtype=dtype))
        return op.add(op.nn.dense(x, w), b)
