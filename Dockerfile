# syntax=docker/dockerfile:1
# ------------------------------------------------------------------------------
# Stage 1: Build binary using official Rust toolchain
# ------------------------------------------------------------------------------
FROM rust:1-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency compilation
COPY Cargo.toml Cargo.lock ./

# Create dummy crates to compile dependencies in a cached layer
RUN mkdir src \
    && echo "fn main() {}" > src/main.rs \
    && touch src/lib.rs \
    && cargo build --release --bin rune-infer \
    && rm -rf src

# Copy real source code and default configuration
COPY src ./src
COPY config ./config

# Invalidate dummy artifacts and build production binary
RUN touch src/main.rs src/lib.rs \
    && cargo build --release --bin rune-infer

# ------------------------------------------------------------------------------
# Stage 2: Minimal hardened runtime image
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Create unprivileged user (UID 10001) to prevent root container breakout
RUN groupadd --gid 10001 rune \
    && useradd --uid 10001 --gid 10001 --home-dir /home/rune --create-home --shell /bin/bash rune

# Install essential runtime dependencies:
# - ca-certificates: TLS certificates for outbound cloud provider proxying
# - curl: required for Docker/Kubernetes healthcheck probes
# - ffmpeg: required for audio/video media extraction and conversions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Prepare application directory owned by rune user
WORKDIR /app
RUN mkdir -p /app/config /app/logs /app/bin && chown -R rune:rune /app /home/rune

# Copy compiled binary from builder stage
COPY --from=builder --chown=rune:rune /build/target/release/rune-infer /app/rune-infer

# Copy default models configuration
COPY --from=builder --chown=rune:rune /build/config/models.json /app/config/models.json

# Switch to unprivileged non-root user
USER rune:rune

# Runtime configuration
ENV RUNE_HOME=/app \
    RUNE_HOST=0.0.0.0 \
    RUNE_PORT=8080

EXPOSE 8080

# Health check probe against dedicated /livez endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/livez || exit 1

# Launch rune-infer in headless daemon mode
ENTRYPOINT ["/app/rune-infer"]
CMD ["--headless"]
