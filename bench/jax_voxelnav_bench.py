"""JAX twin of examples/VoxelNav for the benchmark table. Same terrain
generator, camera, actions, and reward as voxelnav.metal, vectorized with
vmap. Run under jax-metal: python bench/jax_voxelnav_bench.py 4096 [--plain]"""
import time, sys, warnings; warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp, optax, flax.linen as nn
from jax import lax

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 4096
T, EPOCHS, ITERS = 32, 4, 10
W, RAY_COLS, RAY_ROWS = 32, 8, 4
N_RAYS, OBS_DIM, ACT_DIM = RAY_COLS * RAY_ROWS, RAY_COLS * RAY_ROWS + 8, 6
MAX_STEPS, FOV, MAX_DEPTH, EYE, GOAL_RADIUS, START_COBBLE = 250, jnp.pi / 2, 24.0, 1.6, 1.5, 16
PITCHES = jnp.array([-45.0, -20.0, 0.0, 15.0]) * jnp.pi / 180.0
TS = jnp.arange(0.5, MAX_DEPTH, 0.5)  # 47 march samples per ray
print("jax", jax.__version__, jax.devices(), f"N={N} T={T} VoxelNav OBS={OBS_DIM} ACT={ACT_DIM}")

def height_at(height, x, z):
    cx, cz = jnp.floor(x).astype(jnp.int32), jnp.floor(z).astype(jnp.int32)
    inside = (cx >= 0) & (cz >= 0) & (cx < W) & (cz < W)
    return jnp.where(inside, height[jnp.clip(cx, 0, W - 1), jnp.clip(cz, 0, W - 1)].astype(jnp.float32), 0.0), inside

def reset(key):
    k = jax.random.split(key, 8)
    knot = jax.random.uniform(k[0], (5, 5), minval=3.0, maxval=9.0)
    f = jnp.arange(W) / (W - 1) * 4.0
    i = jnp.minimum(f.astype(jnp.int32), 3); t = f - i
    h00 = knot[i[:, None], i[None, :]]; h10 = knot[i[:, None] + 1, i[None, :]]
    h01 = knot[i[:, None], i[None, :] + 1]; h11 = knot[i[:, None] + 1, i[None, :] + 1]
    h = (h00 * (1 - t[:, None]) + h10 * t[:, None]) * (1 - t[None, :]) + (h01 * (1 - t[:, None]) + h11 * t[:, None]) * t[None, :]
    h = jnp.round(h).astype(jnp.int32)
    width = jax.random.randint(k[1], (), 2, 6); left = jax.random.randint(k[2], (), 12, 18); lift = jax.random.randint(k[3], (), 0, 3)
    cx = jnp.arange(W)[:, None]
    h = jnp.where((cx >= left) & (cx < left + width), 0, jnp.where(cx >= left + width, jnp.minimum(h + lift, 15), h))
    u = jax.random.uniform(k[4], (5,))
    x = 2.5 + u[0] * (left - 4); z = 2.5 + u[1] * (W - 5)
    gx = (left + width) + 2.5 + u[2] * (W - 5 - (left + width)); gz = 2.5 + u[3] * (W - 5)
    yaw = -jnp.pi + u[4] * 2 * jnp.pi
    d = jnp.sqrt((gx - x) ** 2 + (gz - z) ** 2)
    return dict(x=x, z=z, yaw=yaw, gx=gx, gz=gz, prevDist=d, steps=jnp.int32(0), cobble=jnp.int32(START_COBBLE), height=h.astype(jnp.uint8))

def observe(s):
    h, _ = height_at(s["height"], s["x"], s["z"])
    ey = h + EYE
    a = s["yaw"] + (jnp.arange(RAY_COLS) / (RAY_COLS - 1) - 0.5) * FOV          # [C]
    dx = jnp.cos(PITCHES)[None, :] * jnp.cos(a)[:, None]                         # [C,R]
    dy = jnp.sin(PITCHES)[None, :] * jnp.ones((RAY_COLS, 1))
    dz = jnp.cos(PITCHES)[None, :] * jnp.sin(a)[:, None]
    px = s["x"] + TS[None, None, :] * dx[..., None]; py = ey + TS[None, None, :] * dy[..., None]; pz = s["z"] + TS[None, None, :] * dz[..., None]
    terrain, inside = height_at(s["height"], px, pz)
    hit = (~inside) | (py <= terrain)
    depth = jnp.min(jnp.where(hit, TS[None, None, :], MAX_DEPTH), axis=-1)       # [C,R]
    gx, gz = s["gx"] - s["x"], s["gz"] - s["z"]
    ca, sa = jnp.cos(s["yaw"]), jnp.sin(s["yaw"])
    gh, _ = height_at(s["height"], s["gx"], s["gz"])
    extra = jnp.stack([(gx * ca + gz * sa) / W, (-gx * sa + gz * ca) / W, (gh - h) / 8.0,
                       jnp.sqrt(gx * gx + gz * gz) / W, sa, ca, s["cobble"] / START_COBBLE, h / 16.0])
    return jnp.concatenate([depth.reshape(-1) / MAX_DEPTH, extra])

def step(s, action, key):
    ca, sa = jnp.cos(s["yaw"]), jnp.sin(s["yaw"])
    here, _ = height_at(s["height"], s["x"], s["z"])
    f = jnp.where(action == 0, 1.0, jnp.where(action == 1, -1.0, jnp.where(action == 4, 2.0, 0.0)))
    moving = f != 0.0
    nx, nz = s["x"] + f * ca, s["z"] + f * sa
    there, inside = height_at(s["height"], nx, nz)
    fell = moving & inside & (there == 0.0)
    can = moving & inside & ((there == 0.0) | (there <= here + 1.0))
    x = jnp.where(can, nx, s["x"]); z = jnp.where(can, nz, s["z"])
    yaw = s["yaw"] + jnp.where(action == 2, -jnp.pi / 6, jnp.where(action == 3, jnp.pi / 6, 0.0))
    yaw = jnp.where(yaw > jnp.pi, yaw - 2 * jnp.pi, jnp.where(yaw < -jnp.pi, yaw + 2 * jnp.pi, yaw))
    # place cobble in front
    pcx, pcz = jnp.floor(s["x"] + ca).astype(jnp.int32), jnp.floor(s["z"] + sa).astype(jnp.int32)
    pin = (pcx >= 0) & (pcz >= 0) & (pcx < W) & (pcz < W)
    pcx, pcz = jnp.clip(pcx, 0, W - 1), jnp.clip(pcz, 0, W - 1)
    cur = s["height"][pcx, pcz].astype(jnp.int32)
    new = jnp.where(cur == 0, here.astype(jnp.int32), jnp.where(cur <= here, cur + 1, cur))
    placing = (action == 5) & pin & (s["cobble"] > 0) & (new != cur)
    height = jnp.where(placing, s["height"].at[pcx, pcz].set(new.astype(jnp.uint8)), s["height"])
    cobble = s["cobble"] - placing.astype(jnp.int32)
    steps = s["steps"] + 1
    d = jnp.sqrt((s["gx"] - x) ** 2 + (s["gz"] - z) ** 2)
    reward = -0.05 + 0.2 * jnp.clip(s["prevDist"] - d, -1.5, 1.5)
    success = d < GOAL_RADIUS
    reward = reward + jnp.where(success, 10.0, 0.0) - jnp.where(fell, 5.0, 0.0)
    done = success | fell | (steps >= MAX_STEPS)
    ns = dict(x=x, z=z, yaw=yaw, gx=s["gx"], gz=s["gz"], prevDist=d, steps=steps, cobble=cobble, height=height)
    obs = observe(ns)
    fresh = reset(key)
    ns = jax.tree_util.tree_map(lambda a, b: jnp.where(done, b, a), ns, fresh)   # auto-reset
    return ns, obs, reward, done

v_reset = jax.vmap(reset); v_step = jax.vmap(step); v_observe = jax.vmap(observe)

class Policy(nn.Module):  # same shape as metalRL: 2 residual LN blocks
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(128)(x)
        for _ in range(2):
            zz = nn.LayerNorm()(h); zz = nn.relu(nn.Dense(512)(zz)); zz = nn.Dense(128)(zz); h = h + zz
        h = nn.LayerNorm()(h)
        return nn.Dense(ACT_DIM)(h), nn.Dense(1)(h).squeeze(-1)
class PolicyPlainMlp(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(128)(x)); h = nn.tanh(nn.Dense(512)(h)); h = nn.tanh(nn.Dense(128)(h))
        return nn.Dense(ACT_DIM)(h), nn.Dense(1)(h).squeeze(-1)
policy = PolicyPlainMlp() if "--plain" in sys.argv else Policy()
key = jax.random.PRNGKey(0)
params = policy.init(key, jnp.zeros((1, OBS_DIM)))
opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-8)); opt_state = opt.init(params)

@jax.jit
def env_only_step(key, state):
    key, k1, k2 = jax.random.split(key, 3)
    acts = jax.random.randint(k1, (N,), 0, ACT_DIM)
    state, obs, r, d = v_step(state, acts, jax.random.split(k2, N))
    return key, state, obs
@jax.jit
def env_only_scan(key, state):
    def body(c, _):
        key, state, _ = env_only_step(*c[:2]); return (key, state, None), None
    (key, state, _), _ = lax.scan(body, (key, state, None), None, length=100)
    return key, state

state = v_reset(jax.random.split(key, N)); obs = v_observe(state)
k, s, o = env_only_step(key, state); o.block_until_ready()
t0 = time.perf_counter()
for _ in range(200): k, s, o = env_only_step(k, s)
o.block_until_ready(); per_step = 200 * N / (time.perf_counter() - t0)
k, s = env_only_scan(key, state); s["x"].block_until_ready()
t0 = time.perf_counter()
for _ in range(5): k, s = env_only_scan(k, s)
s["x"].block_until_ready(); scanned = 500 * N / (time.perf_counter() - t0)
print(f"env only, one jit call per step:  {per_step:12.0f} steps/s")
print(f"env only, 100 steps per jit call: {scanned:12.0f} steps/s")

@jax.jit
def collect(params, key, state, obs):
    def body(c, _):
        key, state, obs = c
        key, k1, k2 = jax.random.split(key, 3)
        logits, v = policy.apply(params, obs)
        act = jax.random.categorical(k1, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(N), act]
        nstate, nobs, r, d = v_step(state, act, jax.random.split(k2, N))
        return (key, nstate, nobs), (obs, act, logp, v, r, d.astype(jnp.float32))
    (key, state, obs), (O, A, LP, V, R, D) = lax.scan(body, (key, state, obs), None, length=T)
    _, next_v = policy.apply(params, obs)
    def gae(carry, x):
        last, nv = carry; r, d, v = x
        nonterm = 1.0 - d; delta = r + 0.99 * nv * nonterm - v
        last = delta + 0.99 * 0.95 * nonterm * last
        return (last, v), last
    _, ADV = lax.scan(gae, (jnp.zeros(N), next_v), (R, D, V), reverse=True)
    return key, state, obs, (O, A, LP, V, ADV, ADV + V)

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
        (params, opt_state), _ = lax.scan(mb, (params, opt_state), perm)
        return (params, opt_state, key), None
    (params, opt_state, key), _ = lax.scan(epoch, (params, opt_state, key), None, length=EPOCHS)
    return params, opt_state, key

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
