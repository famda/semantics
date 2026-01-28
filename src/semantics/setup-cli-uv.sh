#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Input ---
# Expects the full path to the requirements.txt file as the first argument
req_file="$1"

if [ -z "$req_file" ] || [ ! -f "$req_file" ]; then
    echo "Usage: $0 /path/to/module/requirements.txt"
    echo "Error: requirements.txt path not provided or file not found."
    exit 1
fi

# --- Derive Paths ---
module_dir=$(dirname "$req_file")
module_name=$(basename "$module_dir")
module_main_py="$module_dir/main.py" # Path to the module's main.py
venv_path="$module_dir/venv"
venv_python="$venv_path/bin/python"
wrapper_path="/usr/local/bin/$module_name"
cli_wrapper_path="/usr/local/bin/semantics-${module_name}" # Path for the new CLI wrapper
override_file="$module_dir/requirements_override.txt"
lock_file="$module_dir/requirements.lock"

echo "--- Processing module: $module_name found at $module_dir using uv ---"

# Allow forced refresh of lock files when SETUP_CLI_REFRESH_LOCK=1 is present
if [ -n "${SETUP_CLI_REFRESH_LOCK:-}" ]; then
    if [ -f "$lock_file" ]; then
        echo "SETUP_CLI_REFRESH_LOCK detected; removing existing lock file $lock_file"
        rm -f "$lock_file"
    fi
fi

# --- Create Virtual Environment with uv ---
echo "Creating venv at $venv_path using uv (with --system-site-packages to reuse shared CUDA libs)"
uv venv --seed --allow-existing "$venv_path" --python python3

# --- Install Requirements with uv ---
install_source="$req_file"

if [ -f "$lock_file" ]; then
    echo "Detected lock file $lock_file; installing pinned dependencies"
    install_source="$lock_file"
else
    echo "Installing requirements from $req_file into $venv_path"
fi

echo "Installing dependencies with uv pip install..."
uv pip install --python "$venv_python" -r "$install_source"

# Install override requirements on top, if they exist (optional)
if [ -f "$override_file" ]; then
    echo "Installing requirements overrides from $override_file into $venv_path (package by package)"
    while IFS= read -r line || [ -n "$line" ]; do
        spec=$(printf '%s' "$line" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')

        # Skip empty lines and full-line comments
        if [ -z "$spec" ] || [[ "$spec" =~ ^# ]]; then
            continue
        fi

        # Strip inline comments that follow whitespace (keeps URL fragments like #egg=)
        spec=$(printf '%s' "$spec" | sed -E 's/[[:space:]]+#.*$//')

        # Skip if nothing left after stripping
        if [ -z "$spec" ]; then
            continue
        fi

        echo "Installing override spec: $spec"
        uv pip install --python "$venv_python" "$spec"
    done < "$override_file"
fi
# --- End Installation ---

# Persist the exact environment if no lock file was present
if [ ! -f "$lock_file" ]; then
    echo "Freezing dependency versions to $lock_file"
    uv pip freeze --python "$venv_python" > "$lock_file"
fi

nvidia_ld_path=$("$venv_python" - <<'PY'
import os, sysconfig
paths = []
root = os.path.join(sysconfig.get_path("purelib"), "nvidia")
if os.path.isdir(root):
    for current, dirs, files in os.walk(root):
        if os.path.basename(current) == "lib":
            paths.append(os.path.abspath(current))
if paths:
    print(":".join(sorted(set(paths))))
PY
)


# --- No CUDA path detection: rely on Python wheels and system linker ---

# --- Create Wrapper Script ---
echo "Creating wrapper script $wrapper_path"
mkdir -p "$(dirname "$wrapper_path")" # Ensure /usr/local/bin exists

# Use printf for safer script generation
printf '#!/bin/bash\n' > "$wrapper_path"
printf 'set -e\n\n' >> "$wrapper_path"

# Add the final exec command to run python from the venv
printf '# Activate venv and execute python interpreter\n' >> "$wrapper_path"
if [ -n "$nvidia_ld_path" ]; then
    printf 'export LD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$nvidia_ld_path" >> "$wrapper_path"
fi
printf 'exec "%s" "$@"\n' "$venv_python" >> "$wrapper_path"

chmod +x "$wrapper_path"
# --- End Wrapper Script ---


# --- Create CLI Wrapper Script ---
# Check if the module's main.py exists before creating the CLI wrapper
if [ -f "$module_main_py" ]; then
    echo "Creating CLI wrapper script $cli_wrapper_path for $module_main_py"
    mkdir -p "$(dirname "$cli_wrapper_path")" # Ensure /usr/local/bin exists

    # Use printf for safer script generation
    printf '#!/bin/bash\n' > "$cli_wrapper_path"
    printf 'set -e\n\n' >> "$cli_wrapper_path"

    # Add the final exec command to run the module's main.py from the venv
    printf '# Activate venv and execute %s\n' "$module_main_py" >> "$cli_wrapper_path"
    if [ -n "$nvidia_ld_path" ]; then
        printf 'export LD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$nvidia_ld_path" >> "$cli_wrapper_path"
    fi
    # Pass all command-line arguments ("$@") to the main.py script
    printf 'exec "%s" "%s" "$@"\n' "$venv_python" "$module_main_py" >> "$cli_wrapper_path"

    chmod +x "$cli_wrapper_path"
else
    echo "Skipping CLI wrapper creation: $module_main_py not found."
fi
# --- End CLI Wrapper Script ---


echo "--- Finished module: $module_name ---"
exit 0
