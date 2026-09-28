"""Minimal reproduction: jax-metal gather is orders of magnitude slower than
the CPU backend. Run twice: `python bench/jax_metal_gather.py` (Metal) and
`JAX_PLATFORMS=cpu python bench/jax_metal_gather.py`."""
import time, jax, jax.numpy as jnp

B, W, K = 256, 32, 47  # envs, table side, lookups per env
key = jax.random.PRNGKey(0)
table = jax.random.randint(key, (B, W, W), 0, 16).astype(jnp.uint8)
ix = jax.random.randint(jax.random.PRNGKey(1), (B, K), 0, W)
iz = jax.random.randint(jax.random.PRNGKey(2), (B, K), 0, W)

def gather(table, ix, iz):            # table[b, ix[b,k], iz[b,k]]
    return jax.vmap(lambda t, x, z: t[x, z])(table, ix, iz)

def onehot(table, ix, iz):            # same values, no gather: one-hot matmul
    ox = jax.nn.one_hot(ix, W, dtype=jnp.float32)      # [B,K,W]
    oz = jax.nn.one_hot(iz, W, dtype=jnp.float32)
    t = table.astype(jnp.float32)                       # [B,W,W]
    return jnp.einsum("bkx,bxz,bkz->bk", ox, t, oz)

def bench(name, fn):
    f = jax.jit(fn); out = f(table, ix, iz); out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(20): out = f(table, ix, iz)
    out.block_until_ready()
    ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"{name:34s} {ms:8.2f} ms per call   ({B * K / ms * 1e3 / 1e6:8.2f} M lookups/s)")

print("backend:", jax.devices()[0].platform, "jax", jax.__version__)
bench(f"gather  {B}x{K} lookups", gather)
bench(f"one-hot {B}x{K} lookups", onehot)
assert bool(jnp.all(gather(table, ix, iz).astype(jnp.float32) == onehot(table, ix, iz))), "results differ"
print("results identical")
