#!/bin/bash
set -e

# Pick up the same python you’ll use at runtime
PYTHON=$(which python3)
PYTHON_CONFIG=$(which python3-config)

# Compiler and flags
CC=gcc
CFLAGS="-Wall -Wextra -O3 -fPIC -std=c11"
PYTHON_INCLUDES="$($PYTHON_CONFIG --includes)"
PYTHON_LDFLAGS="$($PYTHON_CONFIG --ldflags)"
NUMPY_INCLUDE="$($PYTHON -c 'import numpy; print(numpy.get_include())')"

# Source files and output
SRC_DIR="c"
SRC="$SRC_DIR/main_wrapper.c $SRC_DIR/head_tail_proj.c $SRC_DIR/fast_pcst.c $SRC_DIR/sort.c"
OUTPUT="../sparse_module.so"

echo "Compiling $OUTPUT with $PYTHON"
$CC $CFLAGS $PYTHON_INCLUDES -I$NUMPY_INCLUDE \
    -o $OUTPUT $SRC $PYTHON_LDFLAGS -lm

echo "✅ Built $OUTPUT"
