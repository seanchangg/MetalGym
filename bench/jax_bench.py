import time, sys, warnings; warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp, gymnax, optax, flax.linen as nn
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 4096
T, EPOCHS, ITERS = 32, 4, 10
env, env_params = gymnax.make("CartPole-v1")
print("jax", jax.__version__, jax.devices(), f"N={N} T={T}  policy: 2 residual LN blocks (metalRL shape)")

class PolicyPlainMlp(nn.Module):  #pass --plain to use this one
    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(128)(x)); h = nn.tanh(nn.Dense(512)(h)); h = nn.tanh(nn.Dense(128)(h))
        return nn.Dense(2)(h), nn.Dense(1)(h).squeeze(-1)
#same shape as metalRL: obs -> 128, 2 x (layernorm -> 128->512 relu ->128 -> residual), layernorm, head
class Policy(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(128)(x)
        for _ in range(2):
            z = nn.LayerNorm()(h); z = nn.relu(nn.Dense(512)(z)); z = nn.Dense(128)(z); h = h + z
        h = nn.LayerNorm()(h)
        return nn.Dense(2)(h), nn.Dense(1)(h).squeeze(-1)
policy = PolicyPlainMlp() if "--plain" in sys.argv else Policy()
key = jax.random.PRNGKey(0)
params = policy.init(key, jnp.zeros((1, 4)))
opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-8))
opt_state = opt.init(params)

v_reset = jax.vmap(env.reset, in_axes=(0, None))
v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

# ---- env only
@jax.jit
def env_only_step(key, state):
    key, k1, k2 = jax.random.split(key, 3)
    acts = jax.random.randint(k1, (N,), 0, 2)
    obs, state, r, d, _ = v_step(jax.random.split(k2, N), state, acts, env_params)
    return key, state, obs

@jax.jit
def env_only_scan(key, state):
    def body(c, _):
        key, state, _ = env_only_step(*c[:2]); return (key, state, None), None
    (key, state, _), _ = jax.lax.scan(body, (key, state, None), None, length=100)
    return key, state

keys = jax.random.split(key, N)
obs, state = v_reset(keys, env_params)
k, s, o = env_only_step(key, state); o.block_until_ready()
t0 = time.perf_counter()
for _ in range(200): k, s, o = env_only_step(k, s)
o.block_until_ready(); per_step = 200 * N / (time.perf_counter() - t0)
k, s = env_only_scan(key, state); s.time.block_until_ready()
t0 = time.perf_counter()
for _ in range(5): k, s = env_only_scan(k, s)
s.time.block_until_ready(); scanned = 500 * N / (time.perf_counter() - t0)
print(f"env only, one jit call per step:  {per_step:12.0f} steps/s")
print(f"env only, 100 steps per jit call: {scanned:12.0f} steps/s")

# ---- collect
@jax.jit
def collect(params, key, state, obs):
    def body(c, _):
        key, state, obs = c
        key, k1, k2 = jax.random.split(key, 3)
        logits, v = policy.apply(params, obs)
        act = jax.random.categorical(k1, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(N), act]
        nobs, nstate, r, d, _ = v_step(jax.random.split(k2, N), state, act, env_params)
        return (key, nstate, nobs), (obs, act, logp, v, r, d.astype(jnp.float32))
    (key, state, obs), (O, A, LP, V, R, D) = jax.lax.scan(body, (key, state, obs), None, length=T)
    _, next_v = policy.apply(params, obs)
    def gae(carry, x):
        last, nv = carry; r, d, v = x
        nonterm = 1.0 - d; delta = r + 0.99 * nv * nonterm - v
        last = delta + 0.99 * 0.95 * nonterm * last
        return (last, v), last
    _, ADV = jax.lax.scan(gae, (jnp.zeros(N), next_v), (R, D, V), reverse=True)
    return key, state, obs, (O, A, LP, V, ADV, ADV + V)

# ---- train
def loss_fn(params, o, a, lp_old, adv, ret):
    logits, v = policy.apply(params, o)
    logp = jax.nn.log_softmax(logits)[jnp.arange(N), a]
    ratio = jnp.exp(logp - lp_old); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv).mean()
    vl = 0.5 * ((v - ret) ** 2).mean()
    p = jax.nn.softmax(logits); ent = -(p * jnp.log(p + 1e-8)).sum(-1).mean()
    return pg + 0.5 * vl - 0.01 * ent

@jax.jit
def train(params, opt_state, key, batch):
    O, A, LP, V, ADV, RET = batch
    def epoch(c, _):
        params, opt_state, key = c
        key, k = jax.random.split(key); perm = jax.random.permutation(k, T)
        def mb(c, t):
            params, opt_state = c
            g = jax.grad(loss_fn)(params, O[t], A[t], LP[t], ADV[t], RET[t])
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), None
        (params, opt_state), _ = jax.lax.scan(mb, (params, opt_state), perm)
        return (params, opt_state, key), None
    (params, opt_state, key), _ = jax.lax.scan(epoch, (params, opt_state, key), None, length=EPOCHS)
    return params, opt_state, key

# warm-up compiles
key, state, obs, batch = collect(params, key, state, obs); batch[0].block_until_ready()
params, opt_state, key = train(params, opt_state, key, batch); jax.tree_util.tree_leaves(params)[0].block_until_ready()
ct = tt = 0.0
for it in range(ITERS):
    a = time.perf_counter()
    key, state, obs, batch = collect(params, key, state, obs); batch[0].block_until_ready()
    b = time.perf_counter()
    params, opt_state, key = train(params, opt_state, key, batch); jax.tree_util.tree_leaves(params)[0].block_until_ready()
    c = time.perf_counter(); ct += b - a; tt += c - b
steps = ITERS * T * N
print(f"collect (policy + env + GAE):     {steps / ct:12.0f} steps/s")
print(f"collect + train (4 epochs):       {steps / (ct + tt):12.0f} steps/s   (collect {ct/ITERS:.3f}s, train {tt/ITERS:.3f}s per iter)")
