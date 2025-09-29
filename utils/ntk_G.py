import jax
from jax import random
import jax.numpy as jnp
from neural_tangents import stax
import torch
import numpy as np
from jax import jit

x = torch.randn(18,65,12) # NHWC
adj = torch.randn(65,65)
input_x = torch.matmul(adj, x)
x = x.numpy()
adj = adj.numpy()
dimension_numbers=(((1,), (1,)), ((), ()))
init_fn, apply_fn, kernel_fn = stax.serial(stax.Flatten(1,0),
    stax.DotGeneral(rhs = adj,dimension_numbers=(((1,), (0,)), ((), ())))
                                           )
output = jax.lax.dot_general(adj, x, dimension_numbers)
out = apply_fn((),x)
result = jax.numpy.matmul(adj,x)

print("yes")