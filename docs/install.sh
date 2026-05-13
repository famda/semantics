#!/bin/bash
set -e

# Semantics CLI installer for Linux / macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/famda/semantics/main/docs/install.sh | bash

REPO="famda/semantics"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.semantics/bin}"
IMAGE="famda/semantics:cli-latest"

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
detect_rid() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux*)  os="linux" ;;
        Darwin*) os="osx" ;;
        *)       error "Unsupported OS: $os" ;;
    esac

    case "$arch" in
        x86_64|amd64)  arch="x64" ;;
        arm64|aarch64) arch="arm64" ;;
        *)             error "Unsupported architecture: $arch" ;;
    esac

    echo "${os}-${arch}"
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${GREEN}==>${NC} $1"; }
error() { echo -e "${RED}error:${NC} $1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo -e "${CYAN}${BOLD}  ╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}  ║         Semantics CLI Installer           ║${NC}"
echo -e "${CYAN}${BOLD}  ╚═══════════════════════════════════════════╝${NC}"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    error "Docker is not installed.\n\n  Install Docker first: ${BOLD}https://docs.docker.com/get-docker/${NC}\n\n  Then re-run this installer."
fi

if ! docker info &>/dev/null 2>&1; then
    error "Docker daemon is not running.\n\n  Please start Docker and re-run this installer."
fi

if ! command -v curl &>/dev/null; then
    error "curl is required but not installed."
fi

# ---------------------------------------------------------------------------
# Download gateway binary
# ---------------------------------------------------------------------------
RID="$(detect_rid)"
BINARY_NAME="semantics-${RID}"

info "Downloading Semantics CLI ($RID) ..."

mkdir -p "$INSTALL_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo ".")"
LOCAL_BINARY="${SCRIPT_DIR}/semantics"

# Try local binary first (running from repo checkout with pre-built binary)
if [ -f "$LOCAL_BINARY" ] && [ -x "$LOCAL_BINARY" ]; then
    cp "$LOCAL_BINARY" "${INSTALL_DIR}/semantics"
    chmod +x "${INSTALL_DIR}/semantics"
    info "Installed from local build."
else
    # Download from GitHub Releases
    RELEASE_URL="https://api.github.com/repos/${REPO}/releases/latest"
    DOWNLOAD_URL=$(curl -fsSL "$RELEASE_URL" | grep -o "\"browser_download_url\": *\"[^\"]*${BINARY_NAME}\"" | head -1 | cut -d'"' -f4)

    if [ -z "$DOWNLOAD_URL" ]; then
        error "Could not find ${BINARY_NAME} in latest release."
    fi

    # Download with spinner
    curl -fsSL "$DOWNLOAD_URL" -o "${INSTALL_DIR}/semantics" &
    _dl_pid=$!
    _frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    _i=0
    _start=$SECONDS
    while kill -0 "$_dl_pid" 2>/dev/null; do
        _elapsed=$(( SECONDS - _start ))
        printf '\r  %s Downloading binary ... (%ds)  ' "${_frames:_i%${#_frames}:1}" "$_elapsed"
        _i=$(( _i + 1 ))
        sleep 0.1
    done
    printf '\r%60s\r' ''
    if ! wait "$_dl_pid"; then
        error "Failed to download ${BINARY_NAME}."
    fi

    chmod +x "${INSTALL_DIR}/semantics"
fi

# ---------------------------------------------------------------------------
# Pull Docker image
# ---------------------------------------------------------------------------
info "Pulling container image ..."

docker pull "$IMAGE" > /dev/null 2>&1 &
_pull_pid=$!
_frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
_i=0
_start=$SECONDS
while kill -0 "$_pull_pid" 2>/dev/null; do
    _elapsed=$(( SECONDS - _start ))
    printf '\r  %s Pulling image ... (%ds)  ' "${_frames:_i%${#_frames}:1}" "$_elapsed"
    _i=$(( _i + 1 ))
    sleep 0.1
done
printf '\r%60s\r' ''
if ! wait "$_pull_pid"; then
    error "Failed to pull $IMAGE."
fi

# Create shim scripts for each CLI
for cli in audio video research docs; do
    printf '#!/usr/bin/env bash\nexec "$(dirname "$0")/semantics" %s "$@"\n' "$cli" \
        > "${INSTALL_DIR}/semantics-${cli}"
    chmod +x "${INSTALL_DIR}/semantics-${cli}"
done

info "Installed: semantics, semantics-audio, semantics-video, semantics-research, semantics-docs"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
"${INSTALL_DIR}/semantics" version

# ---------------------------------------------------------------------------
# Add to PATH
# ---------------------------------------------------------------------------
add_to_path() {
    # GitHub Actions: write to GITHUB_PATH
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "$INSTALL_DIR" >> "$GITHUB_PATH"
        info "Added to GITHUB_PATH for this workflow"
        return
    fi

    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"

    local config_file=""
    local path_line=""

    case "$shell_name" in
        bash)
            if [ -f "$HOME/.bashrc" ]; then
                config_file="$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                config_file="$HOME/.bash_profile"
            else
                config_file="$HOME/.bashrc"
            fi
            path_line='export PATH="$HOME/.semantics/bin:$PATH"'
            ;;
        zsh)
            config_file="${ZDOTDIR:-$HOME}/.zshrc"
            path_line='export PATH="$HOME/.semantics/bin:$PATH"'
            ;;
        fish)
            config_file="${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"
            path_line='fish_add_path $HOME/.semantics/bin'
            ;;
        *)
            config_file="$HOME/.profile"
            path_line='export PATH="$HOME/.semantics/bin:$PATH"'
            ;;
    esac

    # Create config directory if needed
    mkdir -p "$(dirname "$config_file")"

    # Skip if already present
    if [ -f "$config_file" ] && grep -q "/.semantics/bin" "$config_file" 2>/dev/null; then
        return
    fi

    echo "" >> "$config_file"
    echo "# Added by Semantics CLI installer" >> "$config_file"
    echo "$path_line" >> "$config_file"

    info "Added to PATH in ${config_file}"
}

if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
    add_to_path
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}  Installation complete!${NC}"
echo ""
echo -e "  Available commands:"
echo -e "    ${BOLD}semantics audio${NC}     — Audio processing"
echo -e "    ${BOLD}semantics video${NC}     — Video analysis"
echo -e "    ${BOLD}semantics research${NC}  — Web research"
echo -e "    ${BOLD}semantics docs${NC}      — Document processing"
echo -e "    ${BOLD}semantics update${NC}    — Update to latest version"
echo ""
echo -e "  ${YELLOW}Restart your terminal${NC} or run:"
echo -e "    source ~/.bashrc  ${DIM}(or ~/.zshrc, etc.)${NC}"
echo ""
