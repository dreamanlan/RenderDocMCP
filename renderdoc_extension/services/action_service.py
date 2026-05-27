"""
Draw call / action operations service for RenderDoc.
"""

import renderdoc as rd

from ..utils import Serializers, Helpers


class ActionService:
    """Draw call / action operations service"""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def get_draw_calls(
        self,
        include_children=True,
        marker_filter=None,
        exclude_markers=None,
        event_id_min=None,
        event_id_max=None,
        only_actions=False,
        flags_filter=None,
    ):
        """
        Get all draw calls/actions in the capture with optional filtering.
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"actions": []}

        def callback(controller):
            root_actions = controller.GetDrawcalls()
            structured_file = controller.GetStructuredFile()
            result["actions"] = Serializers.serialize_actions(
                root_actions,
                structured_file,
                include_children,
                marker_filter=marker_filter,
                exclude_markers=exclude_markers,
                event_id_min=event_id_min,
                event_id_max=event_id_max,
                only_actions=only_actions,
                flags_filter=flags_filter,
            )

        self._invoke(callback)
        return result

    def get_frame_summary(self):
        """
        Get a summary of the current capture frame.
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"summary": None}

        def callback(controller):
            root_actions = controller.GetDrawcalls()
            structured_file = controller.GetStructuredFile()
            api = controller.GetAPIProperties().pipelineType

            # Statistics counters
            stats = {
                "draw_calls": 0,
                "dispatches": 0,
                "clears": 0,
                "copies": 0,
                "presents": 0,
                "markers": 0,
            }
            total_actions = [0]

            def count_actions(actions):
                for action in actions:
                    total_actions[0] += 1
                    flags = action.flags

                    if flags & rd.DrawFlags.Drawcall:
                        stats["draw_calls"] += 1
                    if flags & rd.DrawFlags.Dispatch:
                        stats["dispatches"] += 1
                    if flags & rd.DrawFlags.Clear:
                        stats["clears"] += 1
                    if flags & rd.DrawFlags.Copy:
                        stats["copies"] += 1
                    if flags & rd.DrawFlags.Present:
                        stats["presents"] += 1
                    if flags & (rd.DrawFlags.PushMarker | rd.DrawFlags.SetMarker):
                        stats["markers"] += 1

                    if action.children:
                        count_actions(action.children)

            count_actions(root_actions)

            # Top-level markers
            top_markers = []
            for action in root_actions:
                if action.flags & rd.DrawFlags.PushMarker:
                    child_count = Helpers.count_children(action)
                    top_markers.append({
                        "name": action.name,
                        "event_id": action.eventId,
                        "child_count": child_count,
                    })

            # Resource counts
            textures = controller.GetTextures()
            buffers = controller.GetBuffers()

            result["summary"] = {
                "api": str(api),
                "total_actions": total_actions[0],
                "statistics": stats,
                "top_level_markers": top_markers,
                "resource_counts": {
                    "textures": len(textures),
                    "buffers": len(buffers),
                },
            }

        self._invoke(callback)
        return result["summary"]

    def get_draw_call_details(self, event_id):
        """Get detailed information about a specific draw call"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"details": None, "error": None}

        def callback(controller):
            # Move to the event
            controller.SetFrameEvent(event_id, True)

            action = self.ctx.GetDrawcall(event_id)
            if not action:
                result["error"] = "No action at event %d" % event_id
                return

            structured_file = controller.GetStructuredFile()

            details = {
                "event_id": action.eventId,
                "action_id": action.drawcallId,
                "name": action.name,
                "flags": Serializers.serialize_flags(action.flags),
                "num_indices": action.numIndices,
                "num_instances": action.numInstances,
                "base_vertex": action.baseVertex,
                "vertex_offset": action.vertexOffset,
                "instance_offset": action.instanceOffset,
                "index_offset": action.indexOffset,
            }

            # Output resources
            outputs = []
            for i, output in enumerate(action.outputs):
                if output != rd.ResourceId.Null():
                    outputs.append({"index": i, "resource_id": str(output)})
            details["outputs"] = outputs

            if action.depthOut != rd.ResourceId.Null():
                details["depth_output"] = str(action.depthOut)

            result["details"] = details

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["details"]

    def get_action_timings(
        self,
        event_ids=None,
        marker_filter=None,
        exclude_markers=None,
    ):
        """
        Get GPU timing information for actions.

        Args:
            event_ids: Optional list of specific event IDs to get timings for.
                      If None, returns timings for all actions.
            marker_filter: Only include actions under markers containing this string.
            exclude_markers: Exclude actions under markers containing these strings.

        Returns:
            Dictionary with:
            - available: Whether GPU timing counters are supported
            - unit: Time unit (typically "seconds")
            - timings: List of {event_id, name, duration_seconds, duration_ms}
            - total_duration_ms: Sum of all durations
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None, "error": None}

        def callback(controller):
            # Check if EventGPUDuration counter is available
            counters = controller.EnumerateCounters()
            if rd.GPUCounter.EventGPUDuration not in counters:
                result["data"] = {
                    "available": False,
                    "error": "GPU timing counters not supported on this capture",
                }
                return

            # Get counter description
            counter_desc = controller.DescribeCounter(rd.GPUCounter.EventGPUDuration)

            # Fetch timing data
            counter_results = controller.FetchCounters([rd.GPUCounter.EventGPUDuration])

            # Build event_id to timing map
            timing_map = {}
            target_counter = int(rd.GPUCounter.EventGPUDuration)
            for r in counter_results:
                if r.counter == target_counter:
                    # EventGPUDuration typically returns double
                    # Try to get the value in the most appropriate way
                    val = r.value.d  # double is the standard for duration
                    timing_map[r.eventId] = val

            # Get structured file for action names
            structured_file = controller.GetStructuredFile()
            root_actions = controller.GetDrawcalls()

            # Collect actions to report timings for
            timings = []
            total_duration = [0.0]

            def collect_timings(actions, parent_markers=None):
                if parent_markers is None:
                    parent_markers = []

                for action in actions:
                    action_name = action.name
                    current_markers = parent_markers[:]

                    # Track marker hierarchy
                    is_marker = bool(action.flags & (rd.DrawFlags.PushMarker | rd.DrawFlags.SetMarker))
                    if is_marker:
                        current_markers.append(action_name)

                    # Apply marker filter
                    if marker_filter:
                        marker_path = "/".join(current_markers)
                        if marker_filter.lower() not in marker_path.lower():
                            # Still recurse into children
                            if action.children:
                                collect_timings(action.children, current_markers)
                            continue

                    # Apply exclude filter
                    if exclude_markers:
                        skip = False
                        for exclude in exclude_markers:
                            for m in current_markers:
                                if exclude.lower() in m.lower():
                                    skip = True
                                    break
                            if skip:
                                break
                        if skip:
                            if action.children:
                                collect_timings(action.children, current_markers)
                            continue

                    # Check if we should include this event
                    event_id = action.eventId
                    include = True
                    if event_ids is not None:
                        include = event_id in event_ids

                    if include and event_id in timing_map:
                        duration_sec = timing_map[event_id]
                        duration_ms = duration_sec * 1000.0
                        timings.append({
                            "event_id": event_id,
                            "name": action_name,
                            "duration_seconds": duration_sec,
                            "duration_ms": duration_ms,
                        })
                        total_duration[0] += duration_ms

                    # Recurse into children
                    if action.children:
                        collect_timings(action.children, current_markers)

            collect_timings(root_actions)

            # Sort by event_id
            timings.sort(key=lambda x: x["event_id"])

            result["data"] = {
                "available": True,
                "unit": str(counter_desc.unit),
                "timings": timings,
                "total_duration_ms": total_duration[0],
                "count": len(timings),
            }

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    def get_api_events(
        self,
        event_id_min=None,
        event_id_max=None,
        name_filter=None,
    ):
        """
        Get API-level events (GL/Vulkan/D3D calls) from the structured file.

        Unlike get_draw_calls which only returns action-level events (draw, dispatch,
        clear, marker), this returns the raw API calls recorded in the capture,
        such as glShaderStorageBlockBinding, glBindBufferRange, glUseProgram, etc.

        Args:
            event_id_min: Only include events with chunk index >= this value.
            event_id_max: Only include events with chunk index <= this value.
            name_filter: Only include events whose name contains this string (case-insensitive).

        Returns:
            Dictionary with:
            - events: List of {index, name, num_children, children_summary}
            - count: Number of matching events
            - total_chunks: Total chunks in structured file
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None}

        def callback(controller):
            structured_file = controller.GetStructuredFile()
            chunks = structured_file.chunks

            events = []
            total_chunks = len(chunks)
            name_filter_lower = name_filter.lower() if name_filter else None

            for i in range(total_chunks):
                # Apply range filter
                if event_id_min is not None and i < event_id_min:
                    continue
                if event_id_max is not None and i > event_id_max:
                    break

                chunk = chunks[i]
                chunk_name = chunk.name

                # Apply name filter
                if name_filter_lower and name_filter_lower not in chunk_name.lower():
                    continue

                event_info = {
                    "index": i,
                    "name": chunk_name,
                }

                # Include child data (parameters) summary
                if chunk.NumChildren() > 0:
                    children_summary = []
                    for ci in range(chunk.NumChildren()):
                        child = chunk.GetChild(ci)
                        child_info = {"name": child.name, "type": str(child.type.name)}
                        # Try to get the value for simple types
                        try:
                            if child.type.basetype == rd.SDBasic.Struct:
                                child_info["value"] = "<struct with %d members>" % child.NumChildren()
                            elif child.NumChildren() > 0:
                                # Array or complex type
                                child_info["value"] = "<array with %d elements>" % child.NumChildren()
                            else:
                                # Simple value - try to get string representation
                                val = child.data.str
                                if val:
                                    child_info["value"] = val
                        except Exception:
                            pass
                        children_summary.append(child_info)
                    event_info["parameters"] = children_summary

                events.append(event_info)

            result["data"] = {
                "events": events,
                "count": len(events),
                "total_chunks": total_chunks,
            }

        self._invoke(callback)
        return result["data"]

