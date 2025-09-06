from typing import Dict, Optional, Union
import math
from src.python.configs import *
import numpy as np

class NNUE():
    def __init__(self):
        self.first_layer_stm_weights = np.zeros((HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE), dtype=np.float32)
        self.first_layer_nstm_weights = np.zeros((HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE), dtype=np.float32)
        
        self.first_layer_stm_biases = np.zeros(HIDDEN_LAYER_SIZE, dtype=np.float32)
        self.first_layer_nstm_biases = np.zeros(HIDDEN_LAYER_SIZE, dtype=np.float32)

        self.outpu_layer_weights = np.zeros((1, 2 * HIDDEN_LAYER_SIZE), dtype=np.float32)
        self.output_layer_bias = np.zeros(1, dtype=np.float32)

    def load_weights(self, path_to_bin):
        pass

    def forward(self, x1, x2):
        hiden1 = self.first_layer_stm_weights @ x1 + self.first_layer_stm_biases
        hiden2 = self.first_layer_nstm_weights @ x2 + self.first_layer_nstm_biases

        hiden = np.concatenate((hiden1, hiden2), axis=1)
        hiden = np.clip(hiden, 0, CRelu_Border)

        output = self.outpu_layer_weights @ hiden + self.output_layer_bias
        
        return output


class Loss():
    def __init__(self, X1, X2, Y):
        self.total_size = 2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE + 2*HIDDEN_LAYER_SIZE + 2*HIDDEN_LAYER_SIZE + 1
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def load_model(self, path_to_bin):
        pass

    def _linking(self, params_flat):
        self.A1 = params_flat[0:HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE].\
            reshape(HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE)
        
        self.A2 = params_flat[HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE:2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE].\
            reshape(HIDDEN_LAYER_SIZE, INPUT_VECTOR_SIZE)
        
        self.B1 = params_flat[2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE:2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE+HIDDEN_LAYER_SIZE].\
            reshape(HIDDEN_LAYER_SIZE, 1)
        
        self.B2 = params_flat[2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE+HIDDEN_LAYER_SIZE:2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE+2*HIDDEN_LAYER_SIZE].\
            reshape(HIDDEN_LAYER_SIZE, 1)
        
        self.C  = params_flat[2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE+2*HIDDEN_LAYER_SIZE:2*HIDDEN_LAYER_SIZE*INPUT_VECTOR_SIZE+4*HIDDEN_LAYER_SIZE].\
            reshape(1, 2*HIDDEN_LAYER_SIZE)
        
        self.d  = params_flat[-1].\
            reshape(1,1)

    #TODO def _unlinking

    def __call__(self, X):
        self._linking(X)
        return np.linalg.norm(self.Y - (self.C @ np.clip(np.concatenate((self.A1 @ self.X1 + self.B1, self.A2 @ self.X2 + self.B2), axis=1), 0, CRelu_Border) + self.d))

    def grad(self, X):
        self._linking(X)

        Z = np.concatenate((self.A1 @ self.X1 + self.B1, 
                                            self.A2 @ self.X2 + self.B2), axis=0)  # (2M, B)

        Delta = self.Y - self(X)               # (1, B)

        # dC и dd сразу
        dC = -2 * Delta @ Z.T.clip(0, CRelu_Border)                 # (1, 2M)
        dd = -2 * np.sum(Delta)

        # backprop без хранения mask отдельно
        mask = (Z > 0) & (Z < CRelu_Border)
        del Z       
        back = (self.C.T @ Delta) * mask
        del mask 

        # dA1, dB1
        dA1 = -2 * back[:self.A1.shape[0], :] @ self.X1.T
        dB1 = -2 * np.sum(back[:self.A1.shape[0], :], axis=1, keepdims=True)

        # dA2, dB2
        dA2 = -2 * back[self.A1.shape[0]:, :] @ self.X2.T
        dB2 = -2 * np.sum(back[self.A1.shape[0]:, :], axis=1, keepdims=True)

        # flatten в один вектор (без копий где можно)
        grad_flat = np.empty_like(X)
        offset = 0

        size = self.A1.size
        grad_flat[offset:offset+size] = dA1.ravel(); offset += size

        size = self.A2.size
        grad_flat[offset:offset+size] = dA2.ravel(); offset += size

        size = self.B1.size
        grad_flat[offset:offset+size] = dB1.ravel(); offset += size

        size = self.B2.size
        grad_flat[offset:offset+size] = dB2.ravel(); offset += size

        size = self.C.size
        grad_flat[offset:offset+size] = dC.ravel(); offset += size

        grad_flat[offset] = dd

        return grad_flat

    def __del__(self):
        del self.X1
        del self.X2
        del self.Y
        del self.A1
        del self.A2
        del self.B1
        del self.B2
        del self.C