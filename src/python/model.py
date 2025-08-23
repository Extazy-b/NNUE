from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.python.configs import *

class NNUE(nn.Module):
    """   
    NNUE (Efficiently Updatable Neural Network) for evaluating chess positions.
    
    Architecture:
    - Two sparse binary input vectors (for the active and passive sides)
    - Two separate linear layers for each vector
    - Clipped ReLU activation
    - Concatenation of results
    - Output linear layer
    - Support for weight quantization
    """
    
    def __init__(self, input_size: int = INPUT_VECTOR_SIZE, hidden_size: int = HIDDEN_LAYER_SIZE) -> None:
        """
        Initialize the network.

        Args:
            input_size (int): The size of the input vector (default is in configs)
            hidden_size (int): The size of the hidden layer (default is in configs)

        Returns:
            None
        """

        # Initialize the parent class
        super(NNUE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices and biases for the first layer (separate for each side)
        self.l1_weights = nn.Parameter(torch.empty(2, hidden_size, input_size, dtype=torch.float32))
        self.l1_biases = nn.Parameter(torch.empty(2, hidden_size, dtype=torch.float32))
        
        # Weights and bias for the output layer
        self.l2_weight = nn.Parameter(torch.empty(1, 2 * hidden_size, dtype=torch.float32))
        self.l2_bias = nn.Parameter(torch.empty(1, dtype=torch.float32))
        
        # Scales for quantization (initialized later)
        self.QA = 1.0  # Scale for the first layer
        self.QB = 1.0  # Scale for the output layer
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize weights using the Kaiming method.
        
        The Kaiming method initializes weights with a uniform distribution.
        The arguments are:
        - a: the negative slope of the rectifier used after this layer (only used by 'leaky_relu')
        - mode: either 'fan_in' or 'fan_out'. 'fan_in' preserves the magnitude of the variance of the weights in the forward pass, while 'fan_out' preserves the magnitudes in the backpropagation.
        - nonlinearity: the non-linear function (nn.functional name) used immediately after this linear layer (only applies if mode is 'fan_in'). Default is 'relu'.
        
        We use the defaults for mode and nonlinearity.
        """
        # Initialize weights for the first layer
        for i in range(2):
            nn.init.kaiming_uniform_(self.l1_weights[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.l1_weights[i]) 
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0 
            nn.init.uniform_(self.l1_biases[i], -bound, bound)
        
        # Initialize weights for the output layer
        nn.init.kaiming_uniform_(self.l2_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.l2_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.l2_bias, -bound, bound)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x1: Input vector for the active side (batch_size, input_size)
            x2: Input vector for the passive side (batch_size, input_size)
            
        Returns:
            The evaluation of the position (batch_size)
        """
        # First layer (separate for each input)
        l1_out1 = F.linear(x1, self.l1_weights[0], self.l1_biases[0])  # (batch_size, hidden_size)
        l1_out2 = F.linear(x2, self.l1_weights[1], self.l1_biases[1])  # (batch_size, hidden_size)
        
        # Clipped ReLU activation
        l1_out1 = torch.clamp(l1_out1, min=0, max=CRelu_Border)  # (batch_size, hidden_size)
        l1_out2 = torch.clamp(l1_out2, min=0, max=CRelu_Border)  # (batch_size, hidden_size)
        
        # Concatenate the results
        concatenated = torch.cat([l1_out1, l1_out2], dim=1)  # (batch_size, 2 * hidden_size)
        
        # Output layer
        output = F.linear(concatenated, self.l2_weight, self.l2_bias)  # (batch_size, 1)
        
        return output.squeeze()  # (batch_size)
    
    def quantize_weights(self, QA: Optional[float] = None, QB: Optional[float] = None) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Quantize the weights for inference.
        
        Args:
            QA (float): The scale for the first layer (if None, it is computed automatically)
            QB (float): The scale for the output layer (if None, it is computed automatically)
            
        Returns:
            Dict[str, Union[torch.Tensor, float]]: A dictionary containing the quantized weights and the scales QA and QB
        """
        # Compute the scales if not provided
        if QA is None:
            max_weight1 = torch.max(torch.abs(self.l1_weights))
            max_bias1 = torch.max(torch.abs(self.l1_biases))
            max_val1 = torch.max(max_weight1, max_bias1).item()
            QA = 32767 / max_val1  # 32767 - maximum value for int16
            self.QA = QA  # Save the computed value
        
        if QB is None:
            max_weight2 = torch.max(torch.abs(self.l2_weight))
            max_bias2 = torch.max(torch.abs(self.l2_bias))
            max_val2 = torch.max(max_weight2, max_bias2).item()
            QB = 32767 / max_val2
            self.QB = QB  # Save the computed value
        
        # Quantize the weights
        quantized_l1_weights = (self.l1_weights * QA).round().to(torch.int16)
        quantized_l1_biases = (self.l1_biases * QA).round().to(torch.int16)
        quantized_l2_weight = (self.l2_weight * QB).round().to(torch.int16)
        quantized_l2_bias = (self.l2_bias * QB).round().to(torch.int16)
        
        return {
            'l1_weights': quantized_l1_weights,
            'l1_biases': quantized_l1_biases,
            'l2_weight': quantized_l2_weight,
            'l2_bias': quantized_l2_bias,
            'QA': QA,
            'QB': QB
        }
    def update_accumulator(self, accumulator, features, sign=1):
        """
        Update the accumulator efficiently (for use in the C++ engine).
        
        Args:
            accumulator: The current state of the accumulator
            features: The indices of the updated features
            sign: The direction of the update (1 - add, -1 - remove)
            
        Returns:
            The updated accumulator
        """
        # This function is intended for use in optimized C++ code
        # In the Python version, we just demonstrate the principle
        with torch.no_grad():
            for idx in features:
                accumulator += sign * self.l1_weights[0, :, idx] * self.QA
        return accumulator
    
    def get_optimizer(self, learning_rate=0.001, weight_decay=1e-5):
        """
        Create an optimizer for training the network.
        
        Args:
            learning_rate: The learning rate
            weight_decay: The L2 regularization coefficient
            
        Returns:
            An AdamW optimizer
        """
        return torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    
    def get_lr_scheduler(self, optimizer, step_size=1000, gamma=0.9):
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: The optimizer
            step_size: The step size for the learning rate schedule
            gamma: The learning rate decay factor
            
        Returns:
            A StepLR scheduler
        """
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
