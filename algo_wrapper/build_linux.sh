#!/bin/bash
set -e

# Find the same python you’ll use at runtime
PYTHON=$(which python3)
PYTHON_CONFIG=$(which python3-config)

echo "Using Python interpreter: $PYTHON"

# Compiler and flags
CC=gcc
CFLAGS="-Wall -Wextra -O3 -fPIC -std=c11"
# include dirs
PYTHON_INCLUDES="$($PYTHON_CONFIG --includes)"
NUMPY_INCLUDE="$($PYTHON -c 'import numpy; print(numpy.get_include())')"
# embed pulls in libpython symbols on >=3.8
PYTHON_LDFLAGS="$($PYTHON_CONFIG --ldflags --embed)"

# Source files and output
SRC_DIR="c"
SRC="$SRC_DIR/main_wrapper.c $SRC_DIR/head_tail_proj.c $SRC_DIR/fast_pcst.c $SRC_DIR/sort.c"
OUTPUT="../sparse_module.so"

echo "Compiling $OUTPUT …"
$CC $CFLAGS $PYTHON_INCLUDES -I$NUMPY_INCLUDE -shared \
    $SRC $PYTHON_LDFLAGS -lm

echo "✅ Built $OUTPUT"
