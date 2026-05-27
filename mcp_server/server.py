"""
RenderDoc MCP Server
FastMCP 2.0 server providing access to RenderDoc capture data.
"""

from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from functools import wraps

from .bridge.client import RenderDocBridge, RenderDocBridgeError
from .config import settings

# Initialize FastMCP server
mcp = FastMCP(
    name="RenderDoc MCP Server",
)

# RenderDoc bridge client
bridge = RenderDocBridge(host=settings.renderdoc_host, port=settings.renderdoc_port)


def bridge_tool(fn):
    """
    Wrap a tool function so that bridge/IPC errors are surfaced to the MCP client
    as ToolError (with the real message) instead of a generic Internal Server Error.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RenderDocBridgeError as e:
            raise ToolError(f"RenderDoc bridge error: {e}")
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}")
    return wrapper


@mcp.tool
@bridge_tool
def get_capture_status() -> dict:
    """
    Check if a capture is currently loaded in RenderDoc.
    Returns the capture status and API type if loaded.
    """
    return bridge.call("get_capture_status")


@mcp.tool
@bridge_tool
def get_draw_calls(
    include_children: bool = True,
    marker_filter: str | None = None,
    exclude_markers: list[str] | None = None,
    event_id_min: int | None = None,
    event_id_max: int | None = None,
    only_actions: bool = False,
    flags_filter: list[str] | None = None,
) -> dict:
    """
    Get the list of all draw calls and actions in the current capture.

    Args:
        include_children: Include child actions in the hierarchy (default: True)
        marker_filter: Only include actions under markers containing this string (partial match)
        exclude_markers: Exclude actions under markers containing these strings (list of partial matches)
        event_id_min: Only include actions with event_id >= this value
        event_id_max: Only include actions with event_id <= this value
        only_actions: If True, exclude marker actions (PushMarker/PopMarker/SetMarker)
        flags_filter: Only include actions with these flags (list of flag names, e.g. ["Drawcall", "Dispatch"])

    Returns a hierarchical tree of actions including markers, draw calls,
    dispatches, and other GPU events.
    """
    params: dict[str, object] = {"include_children": include_children}
    if marker_filter is not None:
        params["marker_filter"] = marker_filter
    if exclude_markers is not None:
        params["exclude_markers"] = exclude_markers
    if event_id_min is not None:
        params["event_id_min"] = event_id_min
    if event_id_max is not None:
        params["event_id_max"] = event_id_max
    if only_actions:
        params["only_actions"] = only_actions
    if flags_filter is not None:
        params["flags_filter"] = flags_filter
    return bridge.call("get_draw_calls", params)


@mcp.tool
@bridge_tool
def get_frame_summary() -> dict:
    """
    Get a summary of the current capture frame.

    Returns statistics about the frame including:
    - API type (D3D11, D3D12, Vulkan, etc.)
    - Total action count
    - Statistics: draw calls, dispatches, clears, copies, presents, markers
    - Top-level markers with event IDs and child counts
    - Resource counts: textures, buffers
    """
    return bridge.call("get_frame_summary")


@mcp.tool
@bridge_tool
def find_draws_by_shader(
    shader_name: str,
    stage: Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"] | None = None,
) -> dict:
    """
    Find all draw calls using a shader with the given name (partial match).

    Args:
        shader_name: Partial name to search for in shader names or entry points
        stage: Optional shader stage to search (if not specified, searches all stages)

    Returns a list of matching draw calls with event IDs and match reasons.
    """
    params: dict[str, object] = {"shader_name": shader_name}
    if stage is not None:
        params["stage"] = stage
    return bridge.call("find_draws_by_shader", params)


@mcp.tool
@bridge_tool
def find_draws_by_texture(texture_name: str) -> dict:
    """
    Find all draw calls using a texture with the given name (partial match).

    Args:
        texture_name: Partial name to search for in texture resource names

    Returns a list of matching draw calls with event IDs and match reasons.
    Searches SRVs, UAVs, and render targets.
    """
    return bridge.call("find_draws_by_texture", {"texture_name": texture_name})


@mcp.tool
@bridge_tool
def find_draws_by_resource(resource_id: str) -> dict:
    """
    Find all draw calls using a specific resource ID (exact match).

    Args:
        resource_id: Resource ID to search for (e.g. "ResourceId::12345" or "12345")

    Returns a list of matching draw calls with event IDs and match reasons.
    Searches shaders, SRVs, UAVs, render targets, and depth targets.
    """
    return bridge.call("find_draws_by_resource", {"resource_id": resource_id})


@mcp.tool
@bridge_tool
def get_draw_call_details(event_id: int) -> dict:
    """
    Get detailed information about a specific draw call.

    Args:
        event_id: The event ID of the draw call to inspect

    Includes vertex/index counts, resource outputs, and other metadata.
    """
    return bridge.call("get_draw_call_details", {"event_id": event_id})


@mcp.tool
@bridge_tool
def get_action_timings(
    event_ids: list[int] | None = None,
    marker_filter: str | None = None,
    exclude_markers: list[str] | None = None,
) -> dict:
    """
    Get GPU timing information for actions (draw calls, dispatches, etc.).

    Args:
        event_ids: Optional list of specific event IDs to get timings for.
                   If not specified, returns timings for all actions.
        marker_filter: Only include actions under markers containing this string (partial match).
        exclude_markers: Exclude actions under markers containing these strings.

    Returns timing data including:
    - available: Whether GPU timing counters are supported
    - unit: Time unit (typically "seconds")
    - timings: List of {event_id, name, duration_seconds, duration_ms}
    - total_duration_ms: Sum of all durations
    - count: Number of timing entries

    Note: GPU timing counters may not be available on all hardware/drivers.
    """
    params: dict[str, object] = {}
    if event_ids is not None:
        params["event_ids"] = event_ids
    if marker_filter is not None:
        params["marker_filter"] = marker_filter
    if exclude_markers is not None:
        params["exclude_markers"] = exclude_markers
    return bridge.call("get_action_timings", params)



@mcp.tool
@bridge_tool
def get_api_events(
    event_id_min: int | None = None,
    event_id_max: int | None = None,
    name_filter: str | None = None,
) -> dict:
    """
    Get API-level events (raw GL/Vulkan/D3D calls) from the structured file.

    Unlike get_draw_calls which returns action-level events (draw, dispatch,
    clear, marker), this returns the raw API calls recorded in the capture,
    such as glShaderStorageBlockBinding, glBindBufferRange, glUseProgram, etc.

    Args:
        event_id_min: Only include events with chunk index >= this value
        event_id_max: Only include events with chunk index <= this value
        name_filter: Only include events whose name contains this string (case-insensitive)

    Returns a list of API events with their parameters.
    """
    params: dict[str, object] = {}
    if event_id_min is not None:
        params["event_id_min"] = event_id_min
    if event_id_max is not None:
        params["event_id_max"] = event_id_max
    if name_filter is not None:
        params["name_filter"] = name_filter
    return bridge.call("get_api_events", params)

@mcp.tool
@bridge_tool
def get_shader_info(
    event_id: int,
    stage: Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"],
) -> dict:
    """
    Get shader information for a specific stage at a given event.

    Args:
        event_id: The event ID to inspect the shader at
        stage: The shader stage (vertex, hull, domain, geometry, pixel, compute)

    Returns shader disassembly, constant buffer values, and resource bindings.
    """
    return bridge.call("get_shader_info", {"event_id": event_id, "stage": stage})


@mcp.tool
@bridge_tool
def get_buffer_contents(
    resource_id: str,
    offset: int = 0,
    length: int = 0,
) -> dict:
    """
    Read the contents of a buffer resource.

    Args:
        resource_id: The resource ID of the buffer to read
        offset: Byte offset to start reading from (default: 0)
        length: Number of bytes to read, 0 for entire buffer (default: 0)

    Returns buffer data as base64-encoded bytes along with metadata.
    """
    return bridge.call(
        "get_buffer_contents",
        {"resource_id": resource_id, "offset": offset, "length": length},
    )


@mcp.tool
@bridge_tool
def get_texture_info(resource_id: str) -> dict:
    """
    Get metadata about a texture resource.

    Args:
        resource_id: The resource ID of the texture

    Includes dimensions, format, mip levels, and other properties.
    """
    return bridge.call("get_texture_info", {"resource_id": resource_id})


@mcp.tool
@bridge_tool
def get_texture_data(
    resource_id: str,
    mip: int = 0,
    slice: int = 0,
    sample: int = 0,
    depth_slice: int | None = None,
    event_id: int | None = None,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
) -> dict:
    """
    Read the pixel data of a texture resource.

    Args:
        resource_id: The resource ID of the texture to read
        mip: Mip level to retrieve (default: 0)
        slice: Array slice or cube face index (default: 0)
               For cube maps: 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
        sample: MSAA sample index (default: 0)
        depth_slice: For 3D textures only, extract a specific depth slice (default: None = full volume)
                     When specified, returns only the 2D slice at that depth index
        event_id: If specified, switch replay to this event before reading the texture.
                  CRITICAL for render targets reused across the frame (RT pool):
                  without event_id, the data read is whatever the RT holds at the
                  current/last replay event, which is often zero-cleared or
                  overwritten by a later pass and therefore meaningless.
                  Always pass the event_id of the draw/pass that *wrote* the RT
                  you want to inspect. Default: None = use current/last event.
        x: Crop region left coordinate in pixels at the requested mip level (default: None).
           Must be specified together with y/w/h to enable cropping.
        y: Crop region top coordinate in pixels at the requested mip level (default: None).
        w: Crop region width in pixels (default: None).
        h: Crop region height in pixels (default: None).
           Crop notes:
           - All four of x/y/w/h must be either all None (full mip, default behavior)
             or all set (crop enabled). Partial set is rejected.
           - Crop is NOT supported for compressed/block formats (e.g. BC*, ASTC, ETC);
             read full mip in that case.
           - For 3D textures, crop requires depth_slice to also be specified.
           - Use crop to avoid timeouts/large payloads on big textures (e.g. 4K RTs).
    
    Returns texture pixel data as base64-encoded bytes along with metadata
    including dimensions at the requested mip level and format information.
    When cropping is enabled, width/height in the response are the cropped size,
    and additional fields mip_width/mip_height (original mip size) and
    crop_x/crop_y/crop_w/crop_h (the actual crop region used) are included.
    """
    params = {"resource_id": resource_id, "mip": mip, "slice": slice, "sample": sample}
    if depth_slice is not None:
        params["depth_slice"] = depth_slice
    if event_id is not None:
        params["event_id"] = event_id
    for k, v in (("x", x), ("y", y), ("w", w), ("h", h)):
        if v is not None:
            params[k] = v
    return bridge.call("get_texture_data", params)


@mcp.tool
@bridge_tool
def pick_pixel(
    resource_id: str,
    x: int,
    y: int,
    mip: int = 0,
    slice: int = 0,
    sample: int = 0,
    event_id: int | None = None,
    type_cast: str = "typeless",
) -> dict:
    """
    Read a single pixel value from a texture (cheap, bypasses bulk data transfer).

    Ideal for diagnosing per-pixel issues (e.g. black-screen analysis) when
    the full texture is too large to ship through the MCP channel. Returns
    the pixel in float/uint/sint interpretations so caller can pick whichever
    matches the format semantics.

    Args:
        resource_id: The resource ID of the texture to sample
        x, y: Pixel coordinates within the chosen mip level
        mip: Mip level to sample (default: 0)
        slice: Array slice or cube face index (default: 0)
            For cube maps: 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
        sample: MSAA sample index (default: 0)
        event_id: If specified, switch replay to this event before sampling.
            CRITICAL for render targets reused across the frame (RT pool):
            without event_id, you read whatever the RT holds at the
            current/last replay event, which may be zero-cleared or
            overwritten by a later pass. Always pass the event_id of the
            draw/pass that wrote the RT you want to inspect. Default: None.
        type_cast: How to interpret the raw bits. One of:
            "typeless" (default, use resource's native format),
            "float", "unorm", "snorm", "uint", "sint", "depth", "unorm_srgb".

    Returns dict with float_value/uint_value/int_value (each 4 components),
    plus resource format, mip dimensions, and echoed coordinates.
    """
    params = {
        "resource_id": resource_id,
        "x": x,
        "y": y,
        "mip": mip,
        "slice": slice,
        "sample": sample,
        "type_cast": type_cast,
    }
    if event_id is not None:
        params["event_id"] = event_id
    return bridge.call("pick_pixel", params)


@mcp.tool
@bridge_tool
def save_texture(
    resource_id: str,
    file_path: str,
    file_format: str = "png",
    mip: int = 0,
    slice: int = 0,
    sample: int = 0,
    event_id: int | None = None,
    type_cast: str = "typeless",
    alpha_handling: str = "preserve",
) -> dict:
    """
    Save a texture resource to an image file on disk (server-side write).

    Avoids shipping large texture payloads through the MCP channel. Useful
    when you need the actual pixels for off-line inspection (e.g. black-frame
    analysis, comparing render targets across captures).

    Args:
        resource_id: The resource ID of the texture to save
        file_path: Absolute path on the RenderDoc host where the image will be written
        file_format: Output format. One of:
            "png" (default), "jpg", "bmp", "tga", "hdr", "exr", "dds"
        mip: Mip level to save (default: 0)
        slice: Array slice or cube face index (default: 0)
            For cube maps: 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
        sample: MSAA sample index (default: 0)
        event_id: If specified, switch replay to this event before saving.
            CRITICAL for render targets reused across the frame (RT pool):
            without event_id you save whatever the RT holds at the
            current/last replay event, which may be zero-cleared or
            overwritten by a later pass. Always pass the event_id of the
            draw/pass that wrote the RT you want to inspect. Default: None.
        type_cast: How to interpret the raw bits. One of:
            "typeless" (default, use resource's native format),
            "float", "unorm", "snorm", "uint", "sint", "depth", "unorm_srgb".
        alpha_handling: How to handle alpha channel for formats lacking
            native alpha support (e.g. JPG/BMP). One of:
            "preserve" (default, keep alpha if format supports it),
            "discard", "blend_to_color", "blend_to_checkerboard".

    Returns dict with success flag and saved file path.
    """
    params = {
        "resource_id": resource_id,
        "file_path": file_path,
        "file_format": file_format,
        "mip": mip,
        "slice": slice,
        "sample": sample,
        "type_cast": type_cast,
        "alpha_handling": alpha_handling,
    }
    if event_id is not None:
        params["event_id"] = event_id
    return bridge.call("save_texture", params)



@mcp.tool
@bridge_tool
def get_pipeline_state(event_id: int) -> dict:
    """
    Get the full graphics pipeline state at a specific event.

    Args:
        event_id: The event ID to get pipeline state at

    Returns detailed pipeline state including:
    - Bound shaders with entry points for each stage
    - Shader resources (SRVs): textures and buffers with dimensions, format, slot, name
    - UAVs (RWTextures/RWBuffers): resource details with dimensions and format
    - Samplers: addressing modes, filter settings, LOD parameters
    - Constant buffers: slot, size, variable count
    - Render targets and depth target
    - Viewports and input assembly state
    """
    return bridge.call("get_pipeline_state", {"event_id": event_id})

@mcp.tool
@bridge_tool
def get_postvs(
    event_id: int,
    stage: str = "VSOut",
    instance: int = 0,
    view: int = 0,
    first_vertex: int = 0,
    num_vertices: int = 64,
    parse_position: bool = True,
) -> dict:
    """
    Get Post-VS (post vertex shader) mesh data for a draw call.

    Useful for diagnosing geometry problems such as vertices collapsing to the
    origin (e.g. UBO zero-fill causing LocalToWorld=0 -> all-black BasePass).

    Args:
        event_id: The event ID of the draw call
        stage: Mesh data stage - "VSIn", "VSOut", "GSOut" (default "VSOut")
        instance: Instance index for instanced draws (default 0)
        view: View index for multiview rendering (default 0)
        first_vertex: First vertex to read (default 0)
        num_vertices: Number of vertices to read (default 64, capped by draw size)
        parse_position: If True, parse SV_POSITION/POSITION attribute as float4
                        and include a 'positions' array; also a 'zero_position_count'
                        summary for quick diagnosis (default True)

    Returns:
        dict with mesh metadata (vertex/index buffer info, attribute layout),
        raw vertex bytes (base64) and, when parse_position=True, decoded
        position values plus zero-position statistics.
    """
    return bridge.call(
        "get_postvs",
        {
            "event_id": event_id,
            "stage": stage,
            "instance": instance,
            "view": view,
            "first_vertex": first_vertex,
            "num_vertices": num_vertices,
            "parse_position": parse_position,
        },
    )



@mcp.tool
@bridge_tool
def get_cbuffer_contents(
    event_id: int,
    stage: Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"],
    slot: int = 0,
) -> dict:
    """
    Get the contents of a single constant buffer (UBO) at a given event/stage/slot.

    Targeted accessor with smaller payload than get_pipeline_state/get_shader_info,
    which only return cbuffer metadata as a sub-field. Ideal for diagnosing
    UBO-related issues such as zero-filled BatchedPrimitive UBO (vb2, 16384B)
    causing LocalToWorld=0 -> vertices collapsing to origin -> all-black BasePass.

    Args:
     event_id: The event ID of the draw call to inspect
     stage: Shader stage owning the cbuffer (vertex, hull, domain, geometry, pixel, compute)
     slot: API real binding point of the cbuffer (default: 0). This is the
      same value as cbuffer.slot returned by get_pipeline_state, i.e.
      the GL UBO binding / D3D cb register / Vulkan binding. Pass -1
      to access non-buffer-backed default uniform block (e.g. GL
      $Globals). The internal shader-reflection index is auto-resolved
      via BindpointMapping.

    Returns:
     dict with 'cbuffer' (name, slot, reflection_index, buffer_backed,
     size, variables tree of name/type/value, bound flag, optional
     resource_id/byte_offset/byte_size) and 'error' (str or None).
     'variables' is a serialized tree suitable for direct inspection.
    """
    return bridge.call(
     "get_cbuffer_contents",
        {"event_id": event_id, "stage": stage, "slot": slot},
    )



@mcp.tool
@bridge_tool
def execute_python(
    code: str,
    max_output: int | None = None,
) -> dict:
    """
    Execute arbitrary Python code inside RenderDoc's replay thread.

    Escape hatch for queries not covered by dedicated tools. Runs as a
    privileged plugin (no sandbox) on the RenderDoc host; use responsibly.

    Available names in the user code:
        controller   - ReplayController bound to the current event.
        pyrenderdoc  - qrenderdoc.CaptureContext (high-level UI/capture context).
        rd           - renderdoc module (low-level types, enums, ResourceId, etc.).
        qrd          - qrenderdoc module (may be None if not importable).
        result       - assign here to return a value (default None).

    Args:
        code: Python source string (compiled with exec).
        max_output: Max bytes for stdout / stderr / serialized result.
                    Default 65536 (64 KB). Long outputs are head+tail truncated.

    Returns dict with:
        success: bool - True if exec completed without exception.
        result:  JSON-safe value assigned to `result` (or its string form
                 if not JSON-serializable). May be truncated.
        stdout:  captured stdout text (truncated if oversize).
        stderr:  captured stderr text (truncated if oversize).
        error:   traceback string if exec raised (else absent / empty).
        truncated: {result: bool, stdout: bool, stderr: bool}.

    Notes:
        - Runs via BlockInvoke on the replay thread; long-running or
          infinite-loop code will hang RenderDoc until killed.
        - No timeout is enforced. Keep snippets short and bounded.
        - A capture must be loaded.
    """
    params: dict[str, object] = {"code": code}
    if max_output is not None:
        params["max_output"] = max_output
    return bridge.call("execute_python", params)



@mcp.tool
@bridge_tool
def list_captures(directory: str) -> dict:
    """
    List all RenderDoc capture files (.rdc) in the specified directory.

    Args:
        directory: The directory path to search for capture files

    Returns a list of capture files with their metadata including:
    - filename: The capture file name
    - path: Full path to the file
    - size_bytes: File size in bytes
    - modified_time: Last modified timestamp (ISO format)
    """
    return bridge.call("list_captures", {"directory": directory})


@mcp.tool
@bridge_tool
def open_capture(capture_path: str) -> dict:
    """
    Open a RenderDoc capture file (.rdc).

    Args:
        capture_path: Full path to the capture file to open

    Returns success status and information about the opened capture.
    Note: This will close any currently open capture.
    """
    return bridge.call("open_capture", {"capture_path": capture_path})


def main():
    """Run the MCP server"""
    import logging
    import sys
    import os
    import traceback

    log_dir = r"D:/temp"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    log_path = os.path.join(log_dir, "renderdoc_mcp.log")

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_path,
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    root_logger = logging.getLogger()
    root_logger.info("===== renderdoc-mcp main() entered, pid=%s, argv=%s",
                     os.getpid(), sys.argv)

    def _excepthook(exctype, value, tb):
        root_logger.error("uncaught exception:\n%s",
                          "".join(traceback.format_exception(exctype, value, tb)))
    sys.excepthook = _excepthook

    try:
        mcp.run()
    except BaseException:
        root_logger.exception("mcp.run() raised")
        raise


if __name__ == "__main__":
    main()
