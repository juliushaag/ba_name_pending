# barl_jax

Composable JAX/Flax-NNX actor-critic research framework.

Core APIs:

- `barl_jax.config.run.RunConfig` and `barl_jax.config.model.ModelSpec`
- `barl_jax.policy.Policy`
- `barl_jax.components.ActionSampler`, `Actor`, and `Critic`
- `barl_jax.collection.Collector` and `barl_jax.replay.MultiSeedReplay`
- `barl_jax.checkpoints.save_policy` / `load_policy`
- `barl_jax.trainer.setup` / `train` / `validate`

Optional extras: `logging`, `mujoco`, `dm-control`, `mani-skill`, `humanoid-bench`.
Algorithm defaults live in typed dataclasses beside their implementations.

The repository README documents experiment launching and training-state resume.
