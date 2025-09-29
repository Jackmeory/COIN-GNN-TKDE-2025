import numpy as np
from jax import jit
from neural_tangents import stax
from jax import random
import neural_tangents as nt
from neural_tangents import stax

key1, key2 = random.split(random.PRNGKey(1), 2)
x = random.normal(key1, (20, 32, 32, 3))
y = random.uniform(key1, (20, 10))
_, _, kernel_fn = stax.serial(
    stax.Conv(128, (3, 3)),
    stax.Relu(),
    stax.Conv(256, (3, 3)),
    stax.Relu(),
    stax.Conv(512, (3, 3)),
    stax.Flatten(),
    stax.Dense(10))
k = kernel_fn(x,y)
print("yes")

