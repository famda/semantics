#!/bin/bash
# Reset Platform CLI development environment
# Run this from the Platform directory on the host

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Platform CLI Dev Environment Reset ==="
echo ""
echo "This will stop dev containers and optionally clear cached data."
echo ""

cd "$PLATFORM_DIR"

# Stop containers if running
echo "Stopping dev containers..."
docker compose down 2>/dev/null || true

echo ""
read -p "Also clear workspaces data (.data/platform/workspaces/results)? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing workspaces results..."
    rm -rf .data/platform/workspaces/results/* 2>/dev/null || true
    echo "Workspaces cleared (assets preserved)"
fi

read -p "Rebuild images from scratch? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Rebuilding images (this may take a while)..."
    docker compose build --no-cache
    echo "Images rebuilt"
fi

echo ""
echo "Done! You can now reopen a devcontainer."
