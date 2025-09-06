import os
import numpy as np
from src.python.configs import *
from src.python.dataset import Dataset
from src.python.model import Loss

def grad_destination(F, X0, E1, E2, M, t): # TODO typing
    """
    Gradient descent with backtracking line search.

    Args:
        F: function to minimize (must have .grad() method and be callable)
        X0: initial point (parameter vector)
        E1: gradient norm threshold for early stopping
        E2: tolerance for function value / step change
        M: maximum number of iterations
        t: initial step size
    """
    X1 = np.copy(X0)  # Make a copy of initial point so as not to overwrite it
    for i in range(M):
        t *= 2  # Increase step size before starting backtracking loop
        
        grad = F.grad(X0)  # Compute gradient at current point
        if np.linalg.norm(grad) < E1:
            # If gradient is very small, we assume we are near an optimum
            return X0 
        
        # Backtracking line search:
        # Shrink step size until new point decreases the function value
        while F(X1) - F(X0) >= 0:
            t /= 2  # Halve the step size
            X1 = X0 - t * grad  # Take a smaller step in the opposite direction of gradient

        # Convergence criteria:
        # If both the change in function value and change in parameters are below E2
        if (np.abs(F(X1) - F(X0)) < E2) and (np.linalg.norm(X1 - X0) < E2):
            return X1
        else:
            X0 = X1  # Update current point and continue optimization

    # If maximum iterations reached, return last computed point
    return X1
        

class Trainer():
    def __init__(self, 
                 error: float, 
                 step: float,
                 datasert: Dataset, 
                 max_times: int,
                 reboot: bool, 
                 model_path: str) -> None:
        """
        Trainer class: manages dataset, model initialization, and training.

        Args:
            error: tolerance for optimization (used for both gradient and step size)
            step: initial learning rate (step size)
            max_times: maximum number of optimization iterations
            npz_dir: directory where dataset is stored
            reboot: if True, model parameters are reset (reinitialized)
            model_path: path to save/load model weights in binary format
        """
        
        self.error = error
        self.step = step
        self.max_times = max_times
        self.dataset = datasert # Load dataset from disk
        self.loss = Loss  # Loss function object (not instantiated here yet)
        self.model_path = model_path
        self.model_weight = np.zeros(1)

        if reboot:
            # If reboot flag is set, reinitialize model parameters
            self._reboot()
        else:
            # Otherwise, load model parameters from disk
            self._read_model()
        
    def _reboot(self):
        """
        Reset (reinitialize) all model parameters from scratch
        using He or Xavier initialization where appropriate.
        """
        A1 = self._he_init((HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE))  # first layer weights for X1
        A2 = self._he_init((HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE))  # first layer weights for X2
        B1 = np.zeros((HIDDEN_LAYER_SIZE, 1))  # bias for A1
        B2 = self._he_init((HIDDEN_LAYER_SIZE, 1))  # bias for A2
        C = self._xavier_init((1, 2 * HIDDEN_LAYER_SIZE))  # second layer weights
        d = np.zeros((1, 1))  # output bias
        
        self.model_weight = np.concatenate([ A1.ravel(),  A2.ravel(),  B1.ravel(),  B2.ravel(),  C.ravel(),   d.ravel()])

        del A1
        del A2
        del B1
        del B2
        del C
        del d

        self._write_model()

    def _he_init(self, shape):
        """
        He initialization for ReLU-based networks.
        Random normal distribution scaled by sqrt(2 / fan_in).
        """
        fan_in = shape[1]  # number of input connections
        return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)
    
    def _xavier_init(self, shape):
        """
        Xavier initialization (Glorot uniform).
        Draws from U(-limit, limit), where limit = sqrt(6 / (fan_in + fan_out)).
        """
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).astype(np.float32)

    def train_on_one_batch(self, batch_index):
        """
        Train model on a single batch:
        1. Load current model parameters
        2. Optimize with gradient descent
        3. Save updated model parameters
        """  # load current weights
        try:
            self.model_weight = grad_destination(self.loss, self.model_weight, self.error, self.error, self.max_times, self.step) 
        except Exception as e:
            print(f"Optimization failed: {e}")
        self._write_model()  # save updated weights

    def _read_model(self):
        """
        Read model parameters from binary file.
        (Not implemented yet)
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        self.model_weight = np.fromfile(self.model_path, dtype=np.float32)

    def _write_model(self):
        """
        Write model parameters to binary file.
        (Not implemented yet)
        """
        self.model_weight.astype(np.float32).tofile(self.model_path)

    def end(self):
        """
        End training session.
        (Not implemented yet)
        """
        #TODO backup saving
        


class Validator:
    def __init__(self, validation_dataset) -> None:
        """
        Validator: checks model accuracy on validation dataset.
        """
        self.validation_dataset = validation_dataset

    def check(self):
        """
        Perform validation check.
        Currently a placeholder: always returns True.
        """
        return True
