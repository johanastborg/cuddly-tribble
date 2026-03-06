import jax.numpy as jnp
import jax
import time

def generate_mock_data():
    # Use relative time to avoid float32 precision issues with large unix timestamps
    t = jnp.linspace(0, 600, 600)
    
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    
    # tpu_power: some oscillating power signal
    tpu_power = 100 + 20 * jnp.sin(t * 0.1) + jax.random.normal(k1, shape=(600,))
    
    # subject_c_velocity: negatively correlated with power for "dissonance"
    velocity = 5 - 0.5 * jnp.sin(t * 0.1) + jax.random.normal(k2, shape=(600,))

    # SuperpositionEngine: Simple sine wave with random noise added
    engine_val = 1.0 * jnp.sin(t * 0.1) + 0.1 * jax.random.normal(k1, shape=(600,))

    return {
        'tpu_power': {
            'time': t,
            'value': tpu_power
        },
        'subject_c_velocity': {
            'time': t,
            'value': velocity
        },
        'SuperpositionEngine': {
            'time': t,
            'value': engine_val
        }
    }
