from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class ASGDMT(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, momentum=0.9, yanchi=1.0, use_locking=False, name="ASGDMT"):
        super(ASGDMT, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = momentum
        self._yanchi = yanchi
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None
        self._yanchi_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="momentum")
        self._yanchi_t = ops.convert_to_tensor(self._yanchi, name="yanchi")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "d1", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        yanchi_t = math_ops.cast(self._yanchi_t, var.dtype.base_dtype)

        q = 1.0 / (1.0 - mu_t)
        d = self.get_slot(var, "d1")
        d_t = state_ops.assign(d, yanchi_t / (yanchi_t + q) * d + q / (yanchi_t + q) * (-lr_t * grad))
        var_update = state_ops.assign_add(var, d_t)
        return control_flow_ops.group(*[var_update, d])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
