from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, TYPE_CHECKING

import cv2
from PIL import Image

from .utils.logging import debug_print

if TYPE_CHECKING:
    from config import TilesConfig

__all__ = ["handle"]


def _coerce_frame_indices(indices: Optional[Sequence[object]]) -> List[int]:
    if not indices:
        return []

    coerced: List[int] = []
    for value in indices:
        if isinstance(value, int):
            if value >= 0:
                coerced.append(value)
            continue
        if isinstance(value, float) and value.is_integer() and value >= 0:
            coerced.append(int(value))
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                coerced.append(int(stripped))
    return sorted(set(coerced))


def handle(
    input_file: str,
    output_folder: str,
    config: "TilesConfig | None" = None,
    *,
    frame_indices: Optional[Sequence[object]] = None,
    debug: bool = False,
):
    """Main entry point for tile creation.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: TilesConfig instance or None for defaults.
        frame_indices: List of frame indices to include in tiles.
        debug: Enable verbose debug output.

    Returns:
        Tuple of (tiles_folder, tiles_data).
    """
    return _create(
        input_file,
        output_folder,
        frame_indices,
        columns=config.columns if config else 3,
        rows=config.rows if config else 3,
        final_tile_width=config.final_tile_width if config else None,
        final_tile_height=config.final_tile_height if config else None,
        background_color=config.background_color if config else (0, 255, 0),
        debug=debug,
    )


def _create(
    video_file,
    output_folder,
    frame_indices: Optional[Sequence[object]],
    columns=3,
    rows=3,
    final_tile_width=None,
    final_tile_height=None,
    background_color=(0, 255, 0),
    debug: bool = False,
):

    print("INFO: Creating video tiles")

    # Convert relative paths to absolute paths
    video_file = os.path.abspath(video_file)
    output_folder = os.path.abspath(output_folder)

    output_folder = os.path.join(output_folder, "tiles")
    os.makedirs(output_folder, exist_ok=True)

    selected_indices = _coerce_frame_indices(frame_indices)
    debug_print(f"Selected {len(selected_indices)} frames for tile assembly", debug=debug)

    if not selected_indices:
        print("ERROR: No frames available for tile generation")
        return output_folder, {"tiles": []}

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file: {video_file}")
        return output_folder, {"tiles": []}

    frames_cache: dict[int, Image.Image] = {}
    try:
        # Sort indices to minimize seeking - read in ascending order
        sorted_indices = sorted(selected_indices)
        current_pos = -1
        
        for index in sorted_indices:
            # Only seek if we're not at the next frame position
            if index != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            
            ret, frame = cap.read()
            current_pos = index + 1  # Track where we are after reading
            
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_cache[index] = Image.fromarray(rgb)
    finally:
        cap.release()

    debug_print(f"Decoded {len(frames_cache)} frames for tile assembly", debug=debug)

    available_indices = [idx for idx in selected_indices if idx in frames_cache]

    if not available_indices:
        print("ERROR: Unable to decode any frames for tile generation")
        return output_folder, {"tiles": []}

    if (final_tile_width is None or final_tile_height is None):
        first_image = frames_cache[available_indices[0]]
        final_tile_width = first_image.width
        final_tile_height = first_image.height

    video_abs_path = os.path.abspath(video_file)

    images_per_tile = max(1, columns * rows)
    total_tiles = (len(available_indices) + images_per_tile - 1) // images_per_tile

    tiles_data = {
        "rows": rows,
        "columns": columns,
        "tiles": []
    }

    for tile_index in range(total_tiles):
        debug_print(f"Rendering tile {tile_index + 1} of {total_tiles}", debug=debug)
        # Collect images for this tile
        batch_indices = available_indices[tile_index * images_per_tile : (tile_index + 1) * images_per_tile]

        # Build the "natural" tile first
        row_widths = []
        row_heights = []
        loaded_images = []

        # Break into row chunks
        rows_of_images = [
            batch_indices[r * columns : r * columns + columns] 
            for r in range(rows)
        ]

        for row_imgs in rows_of_images:
            row_list = []
            max_height = 0
            total_width = 0
            for img_file in row_imgs:
                img = frames_cache.get(img_file)
                if img is None:
                    continue
                img_copy = img.copy()
                row_list.append(img_copy)
                total_width += img_copy.width
                if img_copy.height > max_height:
                    max_height = img_copy.height
            row_widths.append(total_width)
            row_heights.append(max_height)
            loaded_images.append(row_list)

        # Natural tile dimensions
        natural_tile_width = max(row_widths) if row_widths else 0
        natural_tile_height = sum(row_heights) if row_heights else 0

        if natural_tile_width == 0 or natural_tile_height == 0:
            # No images, skip
            continue

        # Create the natural tile
        natural_tile = Image.new("RGB", (natural_tile_width, natural_tile_height), background_color)
        y_offset = 0
        for r, row_imgs in enumerate(loaded_images):
            x_offset = 0
            for img in row_imgs:
                natural_tile.paste(img, (x_offset, y_offset))
                x_offset += img.width
            y_offset += row_heights[r]

        # Scale (contain) to fit final_tile_width x final_tile_height
        tile_aspect = natural_tile_width / float(natural_tile_height)
        final_aspect = final_tile_width / float(final_tile_height)

        if tile_aspect > final_aspect:
            scaled_width = final_tile_width
            scaled_height = int(scaled_width / tile_aspect)
        else:
            scaled_height = final_tile_height
            scaled_width = int(scaled_height * tile_aspect)

        scaled_tile = natural_tile.resize((scaled_width, scaled_height), Image.LANCZOS)

        # Center the scaled tile in the final canvas
        final_canvas = Image.new("RGB", (final_tile_width, final_tile_height), background_color)
        offset_x = (final_tile_width - scaled_width) // 2
        offset_y = (final_tile_height - scaled_height) // 2
        final_canvas.paste(scaled_tile, (offset_x, offset_y))

        # Make sure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Save
        final_canvas.save(os.path.join(output_folder, f"tile_{tile_index + 1}.png"))

        tiles_data["tiles"].append({
            "index": tile_index,
            "width": final_canvas.width,
            "height": final_canvas.height,
            "frames": [f"{video_abs_path}#frame_{idx:08d}" for idx in batch_indices]
        })

    # Write JSON file
    with open(os.path.join(output_folder, 'tiles.json'), 'w', encoding='utf-8') as json_file:
        json.dump(tiles_data, json_file, indent=4)

    return output_folder, tiles_data


# def select_frames_by_fps(json_file_path, frames_folder=".", fps=None):

#     with open(json_file_path, "r") as f:
#         data = json.load(f)

#     frames = data.get("frames", [])
#     if not frames:
#         return []

#     # If no FPS provided, return all frame paths in order
#     if fps is None:
#         return [
#             os.path.join(frames_folder, f"{i:08d}.png")
#             for i in range(len(frames))
#         ]

#     # Otherwise, sample the frames according to fps
#     frame_times = [float(frame["pts_time"]) for frame in frames]
#     total_duration = frame_times[-1]
#     selected_frames = []

#     t = 0.0
#     idx = 0
#     step = 1.0 / fps
#     n = len(frames)

#     while t <= total_duration and idx < n:
#         # Move forward until the frame time is >= t
#         while idx < n and frame_times[idx] < t:
#             idx += 1

#         if idx >= n:
#             break

#         filename = os.path.join(frames_folder, f"{idx:08d}.png")
#         selected_frames.append(filename)
#         t += step

#     return selected_frames

# # Example usage
# input_folder = '/workspaces/temp1/temp/frames'
# input_metadata = '/workspaces/temp1/temp/frames/frames.json'
# output_folder = '/workspaces/temp1/temp/frames/tiles'

# # create the output folder if it does not exist
# os.makedirs(output_folder, exist_ok=True)

# columns = 3
# rows = 3
# target_width = 800  # Example target width
# target_height = 500  # Example target height
# create_image_tiles_contain(input_folder, output_folder,input_metadata, columns, rows, fps=4, final_tile_width=target_width, final_tile_height=target_height)
# create_image_tiles_contain(input_folder, output_folder,input_metadata, columns, rows, fps=4)
