# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
LSTM and TreeLSTM defined using Network.
Note how there is no need to write __init__() and forward() as in pytorch -
  only forward(), which we call build_impl(), is needed.
"""
# pylint: disable=invalid-name,missing-docstring,redefined-builtin

from tvm.relay import op, var, Var, Function, Clause, PatternConstructor, PatternVar, Match, const
from tvm.relay import TupleGetItem, Tuple, TensorType, TupleType, If
from network import Network
from network.common import Linear
import numpy as np

def lam(names, func):
    args = [Var(name) for name in names]
    return Function(args, func(*args))

def sigmoid(expr, dtype):
    return op.cast(op.sigmoid(op.cast(expr, dtype="float32")), dtype=dtype)

def tanh(expr, dtype):
    return op.cast(op.tanh(op.cast(expr, dtype="float32")), dtype=dtype)

class LSTMCell(Network):
    """Defining LSTMCell by reusing Linear."""
    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        i = self.input(var("lstmcell_input", shape=(1, input_size), dtype=dtype))
        c = self.input(Var("lstmcell_children", self.p.l(TupleType([t, t]))))
        sum = lam(["x", "y"], lambda x, y: x + y)
        child_h_sum = self.p.foldl(sum,
                                   op.zeros(shape=(1, memory_size), dtype=dtype),
                                   self.p.map(lam(["z"], lambda z: TupleGetItem(z, 1)), c))
        ix, ox, ux = [Linear(input_size=input_size, output_size=memory_size, dtype=dtype)(i) for _ in range(3)]
        ih, oh, uh = [Linear(input_size=memory_size, output_size=memory_size, dtype=dtype)(child_h_sum) for _ in range(3)]
        fx = Linear(input_size=input_size, output_size=memory_size, dtype=dtype)(i)
        fh = Linear(input_size=memory_size, output_size=memory_size, dtype=dtype)
        i, o, u = ix + ih, ox + oh, ux + uh
        i, o, u = sigmoid(i, dtype), sigmoid(o, dtype), tanh(u, dtype)
        def foreach_children(children):
            f = sigmoid(fh(TupleGetItem(children, 1)) + fx, dtype)
            return f * TupleGetItem(children, 0)
        c = self.p.foldl(sum, i * u, self.p.map(lam(["z"], foreach_children), c))
        return Tuple([c, o * tanh(c, dtype)])

class LSTMEncoder(Network):
    """LSTMEncoder is a simple foldl on LSTMCell."""
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.p.l(TensorType(shape=(1, input_size), dtype=dtype))))
        cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype)
        return self.p.foldl(lam(["c", "x"], lambda c, x: cell(x, self.p.cons(c, self.p.nil()))),
                            Tuple([op.zeros(shape=(1, memory_size), dtype=dtype),
                                   op.zeros(shape=(1, memory_size), dtype=dtype)]), l)

class LSTMTransformer(Network):
    """LSTMTransformer is a map_accuml on LSTMCell."""
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.p.l(TensorType(shape=(1, input_size), dtype=dtype))))
        def f(c, x):
            cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype)
            o = cell(x, self.p.cons(c, self.p.nil()))
            return Tuple([o, TupleGetItem(o, 1)])
        res = self.p.map_accuml(lam(["c", "x"], f),
                                Tuple([op.zeros(shape=(1, memory_size), dtype=dtype),
                                       op.zeros(shape=(1, memory_size), dtype=dtype)]),
                                l)
        return Tuple([TupleGetItem(TupleGetItem(res, 0), 1), TupleGetItem(res, 1)])

class TreeLSTM(Network):
    """TreeLSTM is a recursive function that traverse the tree, and reduce using LSTMCell."""
    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        self.ret_type = TupleType([t, t])
        tree_type = self.p.tree(TensorType(shape=(1, input_size), dtype=dtype))
        t = self.input(Var("tlstm_input", tree_type))
        i = Var("i", TensorType(shape=(1, input_size), dtype=dtype))
        c = Var("c", self.p.l(tree_type))
        cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype)
        rose_case = Clause(PatternConstructor(self.p.rose, [PatternVar(i), PatternVar(c)]),
                           cell(i, self.p.map(lam(["x"], self), c)))
        return Match(t, [rose_case])

class BiLSTM(Network):
    """BiLSTM is the zip of two LSTM, with one reversed."""
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.p.l(TensorType(shape=(1, input_size), dtype=dtype))))
        def LSTM(l):
            return LSTMTransformer(input_size=input_size,
                                   memory_size=memory_size,
                                   dtype=dtype)(l)
        fwd = LSTM(l)
        rev = LSTM(self.p.rev(l))
        lhs = op.concatenate([TupleGetItem(fwd, 0), TupleGetItem(rev, 0)], axis=1)
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        x = Var("x", TupleType([t, t])) # cannot infer here
        rhs = self.p.map(Function([x], op.concatenate([TupleGetItem(x, 0),
                                                       TupleGetItem(x, 1)],
                                                      axis=1)),
                         self.p.zip(TupleGetItem(fwd, 1), TupleGetItem(rev, 1)))
        return Tuple([lhs, rhs])
