#!/usr/bin/env python3
"""
JAX/Flax NN architectures used by the QUARTZ training pipeline.

Requires: jax, flax, optax
Docker:   rocm/jax-community:latest (with HSA_OVERRIDE_GFX_VERSION=10.3.0)
PIP:      pip install jax[rocm6] flax optax  (see docs/SETUP.md)
"""
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax.training import train_state
    import optax
    import numpy as np
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if HAS_JAX:
    class SEBlock(nn.Module):
        ch: int
        r: int = 4
        @nn.compact
        def __call__(self, x):
            # Flax Conv/BatchNorm use NHWC by default, so squeeze only spatial axes.
            w = x.mean(axis=(1, 2))
            w = nn.relu(nn.Dense(self.ch // self.r)(w))
            w = nn.sigmoid(nn.Dense(self.ch)(w))
            return x * w[:, None, None, :]

    class ResBlock(nn.Module):
        ch: int
        use_se: bool = False
        @nn.compact
        def __call__(self, x, train=True):
            out = nn.relu(nn.BatchNorm(use_running_average=not train)(x))
            out = nn.Conv(self.ch, (3,3), padding='SAME', use_bias=False)(out)
            out = nn.relu(nn.BatchNorm(use_running_average=not train)(out))
            out = nn.Conv(self.ch, (3,3), padding='SAME', use_bias=False)(out)
            if self.use_se:
                out = SEBlock(self.ch)(out)
            return x + out

    class AlphaZeroJAX(nn.Module):
        """JAX/Flax mirror of PyTorch AlphaZeroNet.
        
        Input:  (batch, C, H, W) — NOTE: we transpose to (batch, H, W, C) internally
        Output: (policy_logits, value)
        """
        board_size: int
        in_ch: int
        n_actions: int
        n_filters: int = 128
        n_blocks: int = 8
        value_hidden: int = 256
        se_blocks: int = 2

        @nn.compact
        def __call__(self, x, train=True):
            # Transpose NCHW → NHWC (JAX convention)
            if x.shape[1] == self.in_ch and x.shape[-1] != self.in_ch:
                x = jnp.transpose(x, (0, 2, 3, 1))
            
            ch = self.n_filters
            n2 = self.board_size * self.board_size

            # Input conv
            h = nn.Conv(ch, (3,3), padding='SAME', use_bias=False)(x)
            h = nn.relu(nn.BatchNorm(use_running_average=not train)(h))

            # Residual tower
            for i in range(self.n_blocks):
                use_se = (i >= self.n_blocks - self.se_blocks)
                h = ResBlock(ch, use_se=use_se)(h, train=train)

            # Policy head
            p = nn.Conv(32, (1,1), use_bias=False)(h)
            p = nn.relu(nn.BatchNorm(use_running_average=not train)(p))
            p = nn.Conv(4, (1,1), use_bias=False)(p)
            p = nn.relu(nn.BatchNorm(use_running_average=not train)(p))
            p = p.reshape(p.shape[0], -1)
            p = nn.Dense(self.n_actions)(p)

            # Value head (global avg pool)
            v = nn.Conv(32, (1,1), use_bias=False)(h)
            v = nn.relu(nn.BatchNorm(use_running_average=not train)(v))
            v = v.mean(axis=(1, 2))
            v = nn.relu(nn.Dense(self.value_hidden)(v))
            v = jnp.tanh(nn.Dense(1)(v)).squeeze(-1)

            return p, v

    # ── Game-specific constructors ──

    def make_gomoku15_jax():
        return AlphaZeroJAX(15, 3, 225, 128, 8, 256, 2)

    def make_chess_jax():
        return AlphaZeroJAX(8, 119, 4096, 128, 10, 256, 2)

    def make_go9_jax():
        return AlphaZeroJAX(9, 17, 82, 128, 10, 256, 2)

    def make_go19_jax():
        return AlphaZeroJAX(19, 17, 362, 192, 15, 256, 4)

    # ── Training utilities ──

    def create_train_state(model, rng, input_shape, lr=0.02, momentum=0.9, wd=1e-4):
        variables = model.init(rng, jnp.ones(input_shape))
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(lr, momentum=momentum),
            optax.add_decayed_weights(wd),
        )
        return train_state.TrainState.create(
            apply_fn=model.apply, params=variables['params'],
            tx=tx, batch_stats=variables.get('batch_stats', {}),
        )

    @jax.jit
    def train_step(state, batch_states, batch_policies, batch_values):
        def loss_fn(params):
            logits, values = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                batch_states, train=True, mutable=['batch_stats'])
            if isinstance(logits, tuple):
                (logits, values), updates = logits  # unpack mutable
            else:
                updates = {}
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            policy_loss = -(batch_policies * log_probs).sum(axis=-1).mean()
            value_loss = jnp.mean((values - batch_values) ** 2)
            return policy_loss + value_loss, (policy_loss, value_loss, updates)
        
        (loss, (p_loss, v_loss, updates)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        if 'batch_stats' in updates:
            state = state.replace(batch_stats=updates['batch_stats'])
        return state, loss, p_loss, v_loss

    def verify_jax():
        """Verify JAX ROCm is working."""
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        x = jnp.ones((2, 3, 15, 15))
        model = make_gomoku15_jax()
        rng = jax.random.PRNGKey(0)
        variables = model.init(rng, x)
        n_params = sum(p.size for p in jax.tree.leaves(variables['params']))
        print(f"Gomoku15 JAX model: {n_params:,} params")
        return True

else:
    def verify_jax():
        print("JAX not installed. Install with:")
        print("  Docker: docker run ... rocm/jax-community:latest")
        print("  PIP:    pip install jax[rocm6] flax optax")
        return False

if __name__ == "__main__":
    verify_jax()
