import os

import platform
import time
import jax
import jax.numpy as jnp
from typing import Optional
import os
from jax.tree_util import tree_map
from copy import deepcopy

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
from dataclasses import dataclass
import tyro
import pandas as pd
from jax.tree_util import tree_map

from transformers import PreTrainedTokenizerFast
import src.jax_rwkv.base_rwkv as rwkv6
from src.jax_rwkv.utils import sample_logits
from src.auto import models, get_model, save, load
from data_loading import _df_to_str, convert_to_nanoseconds, get_data_stream, load_message_df
from constants import MESSAGE_TOKEN_DTYPE_MAP, MESSAGE_TOKEN_TYPES, MambaInferenceArgs
import re
from tokenizers import Tokenizer





@dataclass
class Args:
    model_dir: str
    test_context_file: str
    output_path: str
    device: Optional[str] = None
    strategy: str = "ScanRWKV"
    dtype: str = "float32"
    num_trials: int = 1
    length_per_trial: int = 10000
    temperature: float = 1.0
    top_p: float = 0.9
    num_warmups: int = 1
    context: str = ""




def main():
    args = tyro.cli(Args)

    if args.device is None:
        args.device = 'mps' if platform.system() == 'Darwin' else 'cuda'
    else:
        jax.config.update('jax_platform_name', jax_platforms_map[args.device])
    # Load tokenizer and model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="/app/scripts/lob_tok.json",
        clean_up_tokenization_spaces=False
    )
    model = rwkv6.ScanRWKV()
    params = load(os.path.join(args.model_dir,"params.model"))
    optimizer = load(os.path.join(args.model_dir, "optimizer.model"))
    init_state = load(os.path.join(args.model_dir, "state.model"))

    # Define number of lines to load and convert to a sequence
    n_lines = 100  # Number of lines to load
    n_msgs = 1000    # Number of messages per batch

    #folder = "training_oneDay/data/Flow_10"
    file_name = "AMZN_2017-01-03_24900000_57900000_message_10.csv"
    file_path = os.path.join(args.test_context_file, file_name)
    # Load the DataFrame
    df, removed_indices = load_message_df(file_path, nrows=n_lines)

    if df.empty:
        print("No valid data to process.")
    else:
        # Convert the DataFrame to string format
        context = _df_to_str(df, n_msgs)
    #print(start_tokens)
    # Tokenize context (first 10 lines)
    #context = start_tokens.split('\n')[:10]  # Select first 10 lines as context
    context_text = "\n".join(context)
    #print(context_text)
   

    encoded = tokenizer.encode(context_text)
    if isinstance(tokenizer, Tokenizer):
        encoded = encoded.ids
    ctx_len = len(encoded)
    print(encoded)
   

    # Warm-up phase (optional)
    forward_jit = jax.jit(model.forward)
    forward = lambda x, y: forward_jit(x, y, params)
    for i in range(args.num_warmups):
        print('Warmup', i)
        start_time = time.time()
        _, _ = jax.block_until_ready(forward([0] * ctx_len, init_state))
        _, _ = jax.block_until_ready(forward([0], init_state))
        _, _ = jax.block_until_ready(forward(0, init_state))
        if i == 0:
            print("COMPILE TIME", time.time() - start_time)

    print('\nPreprocessing context')
    start_time = time.time()
    init_out, init_state = jax.block_until_ready(forward(encoded, init_state))
    print(f"Preprocessing time: {time.time() - start_time : .2f} seconds; init_out is {init_out}")
    out_jax = init_out
    if not isinstance(init_out, jnp.ndarray):
        out_jax = init_out.detach().cpu().numpy()

    out_jax = out_jax[ctx_len - 1]


    probs = jax.nn.softmax(jnp.astype(out_jax, jnp.float32), axis=-1) # compute softmax in float (more accurate)

    print(f'\n{context_text}')

    _, indices = jax.lax.top_k(probs, 10) # print top-10 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(token, f'[probability {token_prob:.2%}]')
    
    with open(args.output_path, 'w') as f:
        f.write(context_text + '\n')
        for TRIAL in range(args.num_trials):
            print(f'\n\n--[ Trial {TRIAL} ]-----------------\n', context_text, end="")
            start_time = time.time()
            all_tokens = []
            out_last = 0
            out = init_out[-1].clone()
            # out = init_out
            if init_state is not None:
                state = deepcopy(init_state)#init_state.clone()
            for i in range(args.length_per_trial):
                token = sample_logits(np.asarray(out).astype(np.float64), args.temperature, args.top_p)
                all_tokens += [token]
                try:
                    tmp = tokenizer.decode(all_tokens[out_last:])
                    if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                       # print(tmp, end="", flush=True)
                        formatted_output = tmp.replace('Ċ', '\n')
                        f.write(formatted_output)
                        f.flush()
                        out_last = i + 1
                       
                    
                except Exception as ex:
                    print("INVALID STRING", ex)
                out, state = forward(token, state)
                # print(out.shape)
                out = out[-1]

            print(f'\n--[ Generation Time {time.time() - start_time : .2f} seconds ]-----------------')
            # print(f'--[ Correct: {tokenizer.decode(all_tokens) == answers[TRIAL]} ]-----------------')
        print('\n')
    



if __name__ == "__main__":
    main()
