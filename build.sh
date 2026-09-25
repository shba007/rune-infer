# 1. Correct CUDA 13.3 Root Path (folder, NOT the nvcc.exe file)
export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
export PATH="$CUDA_PATH/bin:$PATH"
export CUDACXX="$CUDA_PATH/bin/nvcc.exe"

# 2. Compile ONLY for RTX 5060 Ti (sm_120)
export CMAKE_CUDA_ARCHITECTURES="120"

# 3. CRITICAL: Disable combinatorial quants
export GGML_CUDA_FA_ALL_QUANTS=OFF
export GGML_CUDA_GRAPHS=ON

# 4. Multi-threading: Tell NVCC (-t 0) and CMake to use all CPU cores
export CMAKE_CUDA_FLAGS="-t 0"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# 5. Clean one last time to purge the old FA_ALL_QUANTS cache, then build:
# cargo clean
cargo run --release --features cuda