import jax
import jaxlib

# Print versions
print("JAX version:", jax.__version__)
print("JAXLIB version:", jaxlib.__version__)

# Check available devices (GPU or CPU)
print("Devices:", jax.devices())

# Check the backend platform
print("Backend platform:", jax.lib.xla_bridge.get_backend().platform)

import jax
import jax.numpy as jnp

# Test JAX with GPU
x = jnp.ones((2, 3))
print(jax.devices())  # Should list the available devices