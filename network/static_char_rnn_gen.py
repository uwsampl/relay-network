from tvm.relay import op, var, Var, Function, Clause, PatternConstructor, PatternVar, Match, const
from tvm.relay import TupleGetItem, Tuple, TensorType, TupleType, If
from network import Network
from common import Linear
import numpy as np
from tvm.relay.quantize import quantize

class CharRNNCell(Network):
    def build_impl(self, input_size, hidden_size, output_size, n_categories=20, n_letters=26):
        category = self.input(var('category', shape=(1, n_categories)))
        inp_topi = self.input(var('input', shape=(), dtype='int32'))
        hidden = self.input(var('hidden', shape=(1, hidden_size)))
        n_letters = const(n_letters)
        one_diag = const(np.diag(np.ones(input_size)).astype('float32'))
        boxed_one = const(np.array([1]).astype('int32'))
        inp = op.take(one_diag, op.multiply(boxed_one, inp_topi), axis=0)
        combined = op.concatenate([op.concatenate([category, inp], axis=1), hidden], axis=1)
        hidden = Linear(input_size=n_categories + input_size + hidden_size, output_size=hidden_size)(combined)
        output = Linear(input_size=n_categories + input_size + hidden_size, output_size=output_size)(combined)
        output_combined = op.concatenate([hidden, output], axis=1)
        output = Linear(input_size=hidden_size + output_size, output_size=output_size)(output_combined)
        topi = op.argmax(output)
        return Tuple([topi, hidden])

class CharRNNGen(Network):
    def build_impl(self, input_size, hidden_size, output_size, n_categories=20, n_letters=26):
        max = self.input(var('max', shape=(), dtype='int32'))
        category = self.input(var('category', shape=(1, n_categories)))
        inp_topi = self.input(var('input', shape=(), dtype='int32'))
        hidden = self.input(var('hidden', shape=(1, hidden_size)))
        fwd_res = CharRNNCell(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_categories=20, n_letters=26)(category, inp_topi, hidden)
        fwd_res_0 = TupleGetItem(fwd_res, 0)
        fwd_res_1 = TupleGetItem(fwd_res, 1)
        else_branch = self.p.cons(fwd_res_1, self(op.subtract(max, const(1)), category, fwd_res_0, fwd_res_1))
        return If(op.equal(max, const(0)), self.p.nil(), else_branch)

class CharRNNGen20(Network):
    def build_impl(self, input_size, hidden_size, output_size, n_categories=20, n_letters=26):
        category = self.input(var('category', shape=(1, n_categories)))
        inp_topi = self.input(var('input', shape=(), dtype='int32'))
        hidden = self.input(var('hidden', shape=(1, hidden_size)))
        crg = CharRNNGen(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_categories=20, n_letters=26)
        return crg(const(20), category, inp_topi, hidden)

x = CharRNNGen20(input_size=22, hidden_size=23, output_size=34)

from tvm.relay.transform import ToANormalForm, PartialEvaluate, ToGraphNormalForm, Sequential

p = Sequential([ToANormalForm(), PartialEvaluate(), ToGraphNormalForm()])
x.mod = p(x.mod)
x.mod["main"] = x.mod["f_0"]
#rnn = quantize(x.mod["main"], x.mod)
print(x.mod["main"])
