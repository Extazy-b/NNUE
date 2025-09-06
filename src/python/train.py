import numpy as np
from src.python.configs import *
from src.python.dataset import Dataset
from src.python.model import Loss

def grad_destination(F, X0, E, E1, E2, M, t): # TODO typing
    """
    Args:
    F: function to minimisation
    X0: initial point
    E1: gradient breakpoint size
    E2: point bias breakpoint size
    M: max count of iterations
    t: step
    """
    X1 = np.copy(X0)
    for i in range(M):
        t *= 2  # Because in the first iteration of each while loop, t is multiplied by 2.
        
        grad = F.grad(X0)
        if np.linalg.norm(grad) < E1:
            return X0 # If the gradient becomes critically small, the process can be interrupted.
        
        while F(X1) - F(X0) >= 0:
            t /= 2 # We adapt the step to ensure that the function is reduced.
            X1 = X0 - t * grad

        if (np.abs(F(X1) - F(X0)) < E2) and (np.linalg.norm(X1 - X0) < E2):
            return X1
        else:
            X0 = X1

    return X1
        

class Trainer():
    def __init__(self, 
                 error: float = 10**(-6), 
                 step: float = 1, 
                 max_times: int = 10**(3), 
                 npz_dir: str = ".data/processed", 
                 reboot: bool = False, 
                 model_path: str = ".model/model.bin") -> None:
        
        self.error = error
        self.step = step
        self.max_times = max_times
        self.dataset = Dataset(npz_dir)
        self.loss = Loss
        self.model_path = model_path

        if reboot:
            self._reboot()
        

    def _reboot(self):
        self.A1 = self._he_init((HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE))
        self.A2 = self._he_init((HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE))
        self.B1 = np.zeros((HIDDEN_LAYER_SIZE, 1))
        self.B2 = self._he_init((HIDDEN_LAYER_SIZE, 1))
        self.C = self._xavier_init((1, 2 * HIDDEN_LAYER_SIZE))
        self.d = np.zeros((1, 1))

    def _he_init(self, shape):
        fan_in = shape[1]  # input size
        return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)
    
    def _xavier_init(self, shape):
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).astype(np.float32)

    def train_on_one_batch(self, batch_index):
        pass
