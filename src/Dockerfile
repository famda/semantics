FROM python:3.13.1-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN mkdir /workspaces && chown root:root /workspaces && chmod 755 /workspaces
RUN mkdir /platform && chown root:root /platform && chmod 755 /platform

# ffmpeg/ffprobe (static) just like the existing image
COPY --from=mwader/static-ffmpeg:8.0 /ffmpeg /usr/local/bin/
COPY --from=mwader/static-ffmpeg:8.0 /ffprobe /usr/local/bin/

# Install system dependencies and uv package manager
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    python3-venv \
    libmagic1 \
    wget \
    # Add PowerShell repo and install (use Ubuntu repo per request) \
    && wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb \
    # Add Microsoft repo for Debian 12 (bookworm) to install PowerShell \
    && wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends powershell \
    # Install ASP.NET Core Runtime 10.0 (used by Worker) \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x dotnet-install.sh \
    && ./dotnet-install.sh --channel 10.0 --runtime aspnetcore --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    # Clean up \
    && rm -f packages-microsoft-prod.deb dotnet-install.sh \
    # Install uv for fast Python package management
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

ARG WORKER_TYPES=""
WORKDIR /platform

COPY ["src/semantics", "temp"]
COPY ["src/semantics/global_helpers.py", "."]
COPY ["src/semantics/setup-agents.sh", "/usr/local/bin/setup-agents.sh"]
# COPY ["src/semantics/setup-cli.sh", "/usr/local/bin/setup-cli.sh"]
COPY ["src/semantics/setup-cli-uv.sh", "/usr/local/bin/setup-cli-uv.sh"]

RUN chmod +x /usr/local/bin/setup-cli-uv.sh /usr/local/bin/setup-agents.sh
RUN /usr/local/bin/setup-agents.sh "${WORKER_TYPES}" ./temp . && rm -rf ./temp

# Create per-module venvs and install their requirements using uv
RUN find /platform -type f -name requirements.txt | while read -r req_file; do /usr/local/bin/setup-cli-uv.sh "$req_file"; done

WORKDIR /workspaces

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "tail -f /dev/null"]
