import time, sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, gymnasium
import pufferlib, pufferlib.vector, pufferlib.emulation, pufferlib.ocean

torch.set_num_threads(8)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
T, EPOCHS, WORKERS = 32, 4, 8

def cartpole_creator(buf=None):
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=gymnasium.make, env_args=["CartPole-v1"], buf=buf)

def make_vec(creator, n, backend, **kw):
    return pufferlib.vector.make(creator, num_envs=n, backend=backend, **kw)

def step_all(vec, actions):
    vec.send(actions)
    out = vec.recv()
    return out[0], out[1], out[2], out[3]  # obs, reward, terminal, truncated

def env_only(vec, n, steps, act_dim):
    vec.reset(seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        step_all(vec, rng.integers(0, act_dim, size=n, dtype=np.int32))
    return steps * n / (time.perf_counter() - t0)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 128), nn.Tanh(), nn.Linear(128, 512), nn.Tanh(), nn.Linear(512, 128), nn.Tanh())
        self.actor = nn.Linear(128, 2); self.critic = nn.Linear(128, 1)
    def forward(self, x):
        h = self.body(x); return self.actor(h), self.critic(h).squeeze(-1)

def ppo(vec, device, iters):
    pol = Policy().to(device); opt = torch.optim.Adam(pol.parameters(), lr=3e-4, eps=1e-8)
    obs_buf = torch.zeros(T, N, 4, device=device); act_buf = torch.zeros(T, N, dtype=torch.long, device=device)
    logp_buf = torch.zeros(T, N, device=device); val_buf = torch.zeros(T, N, device=device)
    rew_buf = torch.zeros(T, N, device=device); done_buf = torch.zeros(T, N, device=device)
    o, _ = vec.reset(seed=0); obs = torch.as_tensor(np.asarray(o), dtype=torch.float32, device=device)
    collect_t = train_t = 0.0
    for it in range(iters):
        a = time.perf_counter()
        for t in range(T):
            with torch.no_grad():
                logits, v = pol(obs); dist = torch.distributions.Categorical(logits=logits); act = dist.sample()
                obs_buf[t] = obs; act_buf[t] = act; logp_buf[t] = dist.log_prob(act); val_buf[t] = v
            o, r, d, tr = step_all(vec, act.to('cpu', torch.int32).numpy())
            obs = torch.as_tensor(np.asarray(o), dtype=torch.float32, device=device)
            rew_buf[t] = torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device)
            done_buf[t] = torch.as_tensor(np.logical_or(d, tr), dtype=torch.float32, device=device)
        with torch.no_grad():
            _, next_v = pol(obs)
            adv = torch.zeros_like(rew_buf); last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + 0.99 * nv * nonterm - val_buf[t]
                last = delta + 0.99 * 0.95 * nonterm * last; adv[t] = last
            ret = adv + val_buf
        if device == 'mps': torch.mps.synchronize()
        b = time.perf_counter()
        for e in range(EPOCHS):
            for t in torch.randperm(T).tolist():
                logits, v = pol(obs_buf[t]); dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_buf[t]); ratio = torch.exp(logp - logp_buf[t])
                a_n = (adv[t] - adv[t].mean()) / (adv[t].std() + 1e-8)
                pg = -torch.min(ratio * a_n, torch.clamp(ratio, 0.8, 1.2) * a_n).mean()
                vl = 0.5 * ((v - ret[t]) ** 2).mean(); ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - 0.01 * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        if device == 'mps': torch.mps.synchronize()
        c = time.perf_counter(); collect_t += b - a; train_t += c - b
    steps = iters * T * N
    return steps / collect_t, steps / (collect_t + train_t), collect_t / iters, train_t / iters

if __name__ == "__main__":
    print(f"PufferLib {pufferlib.__version__}  torch {torch.__version__}  N={N} T={T}")
    # env only
    vec = make_vec(cartpole_creator, 256, pufferlib.vector.Serial)
    print(f"CartPole-v1 (gymnasium) Serial, 256 envs:          {env_only(vec, 256, 200, 2):12.0f} steps/s"); vec.close()
    vec = make_vec(cartpole_creator, N, pufferlib.vector.Multiprocessing, num_workers=WORKERS, batch_size=N)
    print(f"CartPole-v1 (gymnasium) Multiprocessing x{WORKERS}, {N} envs: {env_only(vec, N, 100, 2):12.0f} steps/s")
    # full PPO with the same vec
    for dev in ['cpu', 'mps']:
        cps, full, ct, tt = ppo(vec, dev, 4)
        print(f"PPO on {dev:3s}: collect {cps:12.0f} steps/s   collect+train {full:12.0f} steps/s   (collect {ct:.3f}s, train {tt:.3f}s per iter)")
    vec.close()
    # native C env for reference
    sq = make_vec(pufferlib.ocean.make_squared, 1, pufferlib.vector.Serial, env_kwargs={'num_envs': N})
    print(f"Ocean Squared (native C), {N} envs, Serial:         {env_only(sq, N, 200, 5):12.0f} steps/s"); sq.close()
