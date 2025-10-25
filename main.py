import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

import sys

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 2)

proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

jax.distributed.initialize("localhost:10000", num_procs, proc_id)

assert jax.device_count() == 8

mesh = jax.sharding.Mesh(jax.devices(), ("data",))

global_data = np.arange(32).reshape((8, 4))

sharding = NamedSharding(mesh, P(("data",)))
global_array = jax.device_put(global_data, sharding)
assert global_array.shape == global_data.shape

for shard in global_array.addressable_shards:
    print(f"device {shard.device} has local data {shard.data}")

global_result = jnp.sum(jnp.sin(global_array))
print(f"process={proc_id} got result: {global_result}")
jax.debug.visualize_array_sharding(global_array)
print(jnp.mean(global_array))
jax.distributed.shutdown()
