#!/usr/bin/env bash
# Add micro_engram support to a llama.cpp checkout (Gemma 4): copies src/llama-micro-engram.{h,cpp}
# into it and applies micro-engram.patch (5 small edits). Then build llama.cpp as usual.
#
#   ./llama_cpp/apply.sh /path/to/llama.cpp
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
dst="${1:?usage: apply.sh /path/to/llama.cpp}"
tested=d38bea34227bbf9069d62fdc239c98ab6e05435a

cd "$dst"
if [ ! -f src/models/gemma4.cpp ]; then
    echo "not a llama.cpp checkout (no src/models/gemma4.cpp): $dst" >&2
    exit 1
fi
if grep -q "llama-micro-engram" src/CMakeLists.txt; then
    echo "micro_engram is already applied to $dst"
    exit 0
fi

head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "$head" != "$tested" ]; then
    echo "note: tested on llama.cpp $tested, this checkout is $head (if the patch does not apply, it needs a rebase)"
fi

git apply --check "$here/micro-engram.patch"
git apply "$here/micro-engram.patch"
cp "$here/src/llama-micro-engram.h" "$here/src/llama-micro-engram.cpp" src/

echo "micro_engram applied to $dst. Build as usual, e.g.:"
echo "  cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j --target llama-server"
