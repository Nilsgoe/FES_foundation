#!/usr/bin/env python3
"""Wrapper that patches jax.experimental.layout for orbax compatibility with jax>=0.6."""
import sys
import jax.experimental.layout as _jl

if not hasattr(_jl, 'DeviceLocalLayout'):
    _jl.DeviceLocalLayout = _jl.Layout

from so3lr.cli.so3lr_cli import main
sys.exit(main())
