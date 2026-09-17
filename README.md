set NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor

cargo check --features cuda

cargo run --features cuda

cargo build --features cuda --release