from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class GradientDescentOptimizer(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="SGD"):
        super(GradientDescentOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        var_update = state_ops.assign_sub(var, lr_t * grad)
        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
