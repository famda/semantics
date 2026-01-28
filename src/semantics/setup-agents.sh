#!/bin/bash

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Default source and destination directories
DEFAULT_SOURCE_DIR="platform"
DEFAULT_DEST_DIR="."

# --- Functions ---
usage() {
  echo "Usage: $0 [worker_types_csv] [source_directory] [destination_directory]"
  echo ""
  echo "Arguments:"
  echo "  [worker_types_csv] : Optional. Comma-separated string of worker types (subfolder names) to copy."
  echo "                       If empty or omitted, copies *all* subdirectories found in the source directory."
  echo "                       Example: \"video,audio\""
  echo "  [source_directory] : Optional. The directory containing the worker type subfolders."
  echo "                       Defaults to '${DEFAULT_SOURCE_DIR}'."
  echo "  [destination_directory]: Optional. The directory where worker type folders should be copied."
  echo "                       Defaults to '${DEFAULT_DEST_DIR}' (the current directory)."
  echo ""
  echo "Example (copy specific: video,audio from 'platform' to current dir):"
  echo "  $0 \"video,audio\""
  echo ""
  echo "Example (copy ALL subdirs from 'platform' to './output_workers'):"
  echo "  $0 \"\" ./platform ./output_workers"
  echo "  # OR (omitting first arg also defaults to all)"
  echo "  $0 '' ../my_platform ./output_workers"
  exit 1
}

# --- Argument Parsing ---
WORKER_TYPES_CSV="${1}" # Allow empty string or omitted
SOURCE_DIR="${2:-$DEFAULT_SOURCE_DIR}"
DEST_DIR="${3:-$DEFAULT_DEST_DIR}"

# Validate source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found." >&2
    exit 2
fi

# Ensure destination directory exists (create if not)
mkdir -p "$DEST_DIR"
if [ ! -d "$DEST_DIR" ]; then
    echo "Error: Could not create or find destination directory '$DEST_DIR'." >&2
    exit 3
fi
# Use realpath to handle relative paths like "." cleanly in logs
DEST_DIR=$(realpath "$DEST_DIR")
SOURCE_DIR=$(realpath "$SOURCE_DIR")

echo "INFO: Source Directory: $SOURCE_DIR"
echo "INFO: Destination Directory: $DEST_DIR"

# --- Logic ---
# Determine which types we actually need to copy
TYPES_TO_COPY=""
REQUESTED_SPECIFIC_TYPES=0 # Flag to track if user provided a specific list

if [ -z "${WORKER_TYPES_CSV}" ]; then
    # --- Auto-detect subdirectories ---
    echo "INFO: Worker types list is empty or not provided. Detecting all immediate subdirectories in '$SOURCE_DIR'."
    DETECTED_TYPES=""
    # Use find to get immediate subdirectories, then extract basename
    # Use process substitution `< <(...)` for safer reading than piping to `while read` in some shells
    while IFS= read -r dir_path; do
        # Get the base directory name
        type_name=$(basename "$dir_path")
        # Optional: Add filtering here if needed (e.g., skip hidden folders)
        # if [[ "$type_name" == .* ]]; then continue; fi
        DETECTED_TYPES="${DETECTED_TYPES}${type_name} "
    done < <(find "$SOURCE_DIR" -maxdepth 1 -mindepth 1 -type d) # Find only immediate subdirs

    # Remove potential trailing space and normalize spaces
    TYPES_TO_COPY=$(echo "$DETECTED_TYPES" | xargs)

    if [ -z "$TYPES_TO_COPY" ]; then
        echo "WARNING: No subdirectories found in '$SOURCE_DIR' to copy by default."
    else
         echo "INFO: Automatically detected types to copy: ${TYPES_TO_COPY}"
    fi
    # --- End Auto-detect ---
else
    # --- Use provided list ---
    echo "INFO: Worker types provided: '${WORKER_TYPES_CSV}'. Copying specified types."
    # Replace commas with spaces for iteration
    TYPES_TO_COPY=$(echo "${WORKER_TYPES_CSV}" | sed 's/,/ /g')
    REQUESTED_SPECIFIC_TYPES=1
    # --- End Use provided list ---
fi

# --- Execution ---
echo "INFO: Final list of types to process for copying: ${TYPES_TO_COPY}"
COPIED_COUNT=0
FAILED_COUNT=0

# Proceed only if there are types to copy
if [ -n "$TYPES_TO_COPY" ]; then
    # Loop through the determined types (either detected or specified)
    for type in ${TYPES_TO_COPY}; do
        # Trim potential whitespace around the type name (handled by xargs earlier, but good practice)
        type=$(echo "$type" | xargs)
        # Skip if type name ended up empty after trimming (unlikely now but safe)
        if [ -z "$type" ]; then continue; fi

        SOURCE_PATH="${SOURCE_DIR}/${type}"
        DEST_PATH="${DEST_DIR}/${type}"

        # Check if the source directory actually exists (redundant as find already found them for default case, but safe)
        if [ -d "${SOURCE_PATH}" ]; then
            echo "INFO: Copying worker type '${type}' from ${SOURCE_PATH} to ${DEST_PATH}"
            # Copy the directory recursively, preserving metadata if possible
            cp -a "${SOURCE_PATH}" "${DEST_PATH}" || { echo "Error: Failed to copy ${SOURCE_PATH}" >&2; FAILED_COUNT=$((FAILED_COUNT + 1)); }
            # Check if cp succeeded before incrementing count
             if [ $? -eq 0 ]; then
               COPIED_COUNT=$((COPIED_COUNT + 1));
             fi
        else
            # This should only happen if a specific type was requested but doesn't exist
            if [ $REQUESTED_SPECIFIC_TYPES -eq 1 ]; then
                echo "WARNING: Requested worker type folder '${SOURCE_PATH}' not found. Skipping." >&2
                FAILED_COUNT=$((FAILED_COUNT + 1)) # Count missing requested types as failures
            else
                # Should not happen in default case because `find` located the dir. Log if it does.
                 echo "ERROR: Directory '${SOURCE_PATH}' found by 'find' but now missing? Skipping." >&2
                 FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
        fi
    done
else
    echo "INFO: No types specified or detected, nothing to copy."
fi

# --- Validation & Exit ---
echo "INFO: Finished processing."
echo "INFO: Successfully copied $COPIED_COUNT worker type(s)."

if [ $FAILED_COUNT -gt 0 ]; then
    echo "ERROR: $FAILED_COUNT problem(s) encountered (requested type missing or copy failed)." >&2
    exit 4 # Exit with error code if problems occurred
fi

# Differentiate between requesting specific types and finding none vs defaulting and finding none
if [ $COPIED_COUNT -eq 0 ] && [ $REQUESTED_SPECIFIC_TYPES -eq 1 ] && [ -n "$TYPES_TO_COPY" ]; then
    # User requested specific types, but none existed or copied successfully.
    echo "WARNING: None of the requested worker types (${WORKER_TYPES_CSV}) were found or copied successfully." >&2
    # Exit with error because user request failed
    exit 5
elif [ $COPIED_COUNT -eq 0 ] && [ $REQUESTED_SPECIFIC_TYPES -eq 0 ]; then
     # Default case ran but found no directories or failed to copy them (FAILED_COUNT > 0 handled above)
     # This warning is covered by the 'No subdirectories found' message earlier or the FAILED_COUNT check.
     : # No additional specific warning needed here for the default case finding nothing.
fi

echo "INFO: Script completed."
exit 0 # Success