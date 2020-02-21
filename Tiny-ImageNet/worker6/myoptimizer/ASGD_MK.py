from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class ASGDMK(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, yanchi=1.0, count=1.0, use_locking=False, name="ASGDMK"):
        super(ASGDMK, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._yanchi = yanchi
        self._count = count
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._yanchi_t = None
        self._count_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._yanchi_t = ops.convert_to_tensor(self._yanchi, name="yanchi")
        self._count_t = ops.convert_to_tensor(self._count, name="count")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "d1", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        yanchi_t = math_ops.cast(self._yanchi_t, var.dtype.base_dtype)
        count_t = math_ops.cast(self._count_t, var.dtype.base_dtype)

        d = self.get_slot(var, "d1")
        d_t = state_ops.assign(d, (1.0 - (1.0 / count_t)) * d - (1.0 / count_t) * lr_t * grad)
        var_update = state_ops.assign_add(var, d_t)
        return control_flow_ops.group(*[var_update, d])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
