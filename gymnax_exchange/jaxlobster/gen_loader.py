import os

import platform
import time
import jax
from pathlib import Path
import jax.numpy as jnp
from typing import Optional
import os
import sys
from jax.tree_util import tree_map
from copy import deepcopy
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
)

#jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
#jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
#jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import numpy as np
#from dataclasses import dataclass
#import tyro
import pandas as pd
from jax.tree_util import tree_map

from transformers import PreTrainedTokenizerFast
import jax_rwkv.src.jax_rwkv.base_rwkv as rwkv6
from jax_rwkv.src.jax_rwkv.utils import sample_logits
from jax_rwkv.src.auto import models, get_model, save, load
import re
from tokenizers import Tokenizer

class GenLoader:
    """
    Stateful generator for LOBSTER-style messages with context-aware generation.
    
    Attributes
    ----------
    model : callable
        Function that generates next messages given (context, ob_state)
    context : jax.Array
        Running message context (n_messages, 8)
    ob_state : jax.Array
        Current order book state (maintained for compatibility)
    n_messages : int
        Number of messages to generate per step (default 100)
    """
    
    def __init__(self,
                 model,
                 initial_context,
                 initial_ob_state,
                 n_messages=100):
        self.model = model
        self.context = initial_context
        self.ob_state = initial_ob_state
        self.n_messages = n_messages

    def generate_step(self):
        """
        Generate next batch of messages and update internal state.
        
        Returns:
            new_messages (jax.Array): (n_messages, 8) shaped array
            ob_state (jax.Array): Current order book state (pass-through)
        """
        # Generate next message batch from current context
        new_messages = self.model(self.context, self.ob_state,self.n_messages)
        
        # Ensure correct number of messages
        new_messages = new_messages[:self.n_messages]
        
        # Update generation context (sliding window would be better for production)
        self.context = jnp.concatenate([self.context, new_messages], axis=0)
        
        return new_messages, self.ob_state

    def update_ob_state(self, new_ob_state):
        """Update order book state (for downstream compatibility)"""
        self.ob_state = new_ob_state

    @staticmethod
    def dummy_model(context, ob_state,n_messages):
        """
        Example model implementation (replace with RWKV).
        Returns (100, 8) shaped random messages.
        """
        return jax.random.randint(
            jax.random.PRNGKey(0),
            (100, 8),
            minval=0,
            maxval=1000
        )
def messages_to_text(messages: jnp.ndarray) -> str:
    """Convert (N, 8) message array to tokenizable text"""
    return ' '.join([','.join(map(str, msg)) for msg in messages])

def text_to_messages(text: str) -> jnp.ndarray:
    """Convert generated text back to (N, 8) message array"""
    return jnp.array([list(map(float, line.split(','))) 
                     for line in text.split()]).reshape(-1, 8)
class RWKVWrapper:
    def __init__(self):
        self.model= rwkv6.ScanRWKV()
        # Get absolute paths using Pathlib
        base_dir = Path.home() /"AlphaTrade"/"jax_rwkv"
        
        # Verify and load tokenizer
        tokenizer_path = base_dir / "scripts" / "lob_tok.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
            
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            clean_up_tokenization_spaces=False
        )

        # Verify and load model components
        model_dir = base_dir / "bptt_rwkv_6g0.1B"/"bptt_rwkv_6g0.1B"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")

        #self.params = load(os.path.join(model_dir, "params.model"))
        self.optimizer = load(os.path.join(model_dir, "optimizer.model"))
        self.init_state = load(os.path.join(model_dir, "state.model"))
        
        self.temperature = 1.0
        self.top_p = 0.9
        self.forward_jit = jax.jit(self.model.forward)
        

    def __call__(self, context: jnp.ndarray, ob_state, n_messages: int):
        # Convert numeric context to tokenizable text
        context_text = messages_to_text(context)
        
        # Tokenize with proper handling
        encoded = self.tokenizer.encode(context_text).ids
        if len(encoded) == 0:
            encoded = [self.tokenizer.pad_token_id]

        # Run model forward pass
        out, state = self.forward_jit(encoded, self.init_state, self.params)
        
        # Generation loop
        all_tokens = []
        current_state = state
        for _ in range(n_messages):
            # Sample next token
            probs = jax.nn.softmax(out[-1].astype(jnp.float32), axis=-1)
            token = sample_logits(probs, self.temperature, self.top_p)
            all_tokens.append(int(token))
            
            # Forward pass with new token
            out, current_state = self.forward_jit(
                jnp.array([token]), current_state, self.params
            )

        # Decode and convert back to messages
        generated_text = self.tokenizer.decode(all_tokens)
        return text_to_messages(generated_text)

# Usage Example
if __name__ == "__main__":
  
    # Initialize with proper context shape
    initial_context = jnp.zeros((100, 8))  # Match model's expected context window
    print(Path.home())
    # Create loader with warmup
    loader = GenLoader(
        model=GenLoader.dummy_model,
        initial_context=initial_context,
        initial_ob_state=jnp.zeros((2, 10)),
        n_messages=100
    )

    # Generate with progress monitoring
    for step in range(10):
        messages, _ = loader.generate_step()
        print(f"Step {step+1} generated {messages.shape[0]} messages")
        #loader.update_ob_state(calculate_new_ob_state(messages))  # Implement this



# Usage
#rwkv_model = RWKVWrapper()
#loader = GenLoader(model=rwkv_model, ...)