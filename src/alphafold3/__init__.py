# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""An implementation of the inference pipeline of AlphaFold 3."""

import io
import os
import subprocess as sp

import pandas as pd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"


def _resolve_xla_flags():
  flags = "--xla_cpu_multi_thread_eigen=true --xla_gpu_enable_triton_gemm=false"
  if extra_flags := os.environ.get("XLA_FLAGS"):
    flags += f" {extra_flags}"

  try:
    out = sp.run(
      ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"],
      check=True,
      stdout=sp.PIPE,
      text=True,
    )
    cap = pd.read_csv(io.StringIO(out.stdout))["compute_cap"].min()
    if cap < 8.0:
      flags += " --xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
  except FileNotFoundError:
    pass

  os.environ["XLA_FLAGS"] = flags


_resolve_xla_flags()
