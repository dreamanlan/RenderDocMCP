"""
Pipeline state service for RenderDoc.
"""

import renderdoc as rd
import base64
import struct


from ..utils import Parsers, Serializers, Helpers


class PipelineService:
    """Pipeline state service"""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def get_shader_info(self, event_id, stage):
        """Get shader information for a specific stage"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"shader": None, "error": None}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)

            pipe = controller.GetPipelineState()
            stage_enum = Parsers.parse_stage(stage)

            shader = pipe.GetShader(stage_enum)
            if shader == rd.ResourceId.Null():
                result["error"] = "No %s shader bound" % stage
                return

            entry = pipe.GetShaderEntryPoint(stage_enum)
            reflection = pipe.GetShaderReflection(stage_enum)

            shader_info = {
                "resource_id": str(shader),
                "entry_point": entry,
                "stage": stage,
            }

            # Get disassembly
            try:
                targets = controller.GetDisassemblyTargets(True)
                if targets:
                    disasm = controller.DisassembleShader(
                        pipe.GetGraphicsPipelineObject(), reflection, targets[0]
                    )
                    shader_info["disassembly"] = disasm
            except Exception as e:
                shader_info["disassembly_error"] = str(e)

            # Get constant buffer info
            if reflection:
                shader_info["constant_buffers"] = self._get_cbuffer_info(
                    controller, pipe, reflection, stage_enum
                )
                shader_info["resources"] = self._get_resource_bindings(reflection)

            result["shader"] = shader_info

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["shader"]

    def get_pipeline_state(self, event_id):
        """Get full pipeline state at an event"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"pipeline": None, "error": None}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)

            pipe = controller.GetPipelineState()
            api = controller.GetAPIProperties().pipelineType

            pipeline_info = {
                "event_id": event_id,
                "api": str(api),
            }

            # Shader stages with detailed bindings
            stages = {}
            stage_list = Helpers.get_all_shader_stages()
            for stage in stage_list:
                shader = pipe.GetShader(stage)
                if shader != rd.ResourceId.Null():
                    stage_info = {
                        "resource_id": str(shader),
                        "entry_point": pipe.GetShaderEntryPoint(stage),
                    }

                    reflection = pipe.GetShaderReflection(stage)

                    stage_info["resources"] = self._get_stage_resources(
                        controller, pipe, stage, reflection
                    )
                    stage_info["uavs"] = self._get_stage_uavs(
                        controller, pipe, stage, reflection
                    )
                    stage_info["samplers"] = self._get_stage_samplers(
                        pipe, stage, reflection
                    )
                    stage_info["constant_buffers"] = self._get_stage_cbuffers(
                        controller, pipe, stage, reflection
                    )

                    stages[str(stage)] = stage_info

            pipeline_info["shaders"] = stages

            # Viewport and scissor
            try:
                vp_scissor = pipe.GetViewportScissor()
                if vp_scissor:
                    viewports = []
                    for v in vp_scissor.viewports:
                        viewports.append(
                            {
                                "x": v.x,
                                "y": v.y,
                                "width": v.width,
                                "height": v.height,
                                "min_depth": v.minDepth,
                                "max_depth": v.maxDepth,
                            }
                        )
                    pipeline_info["viewports"] = viewports
            except Exception:
                pass

            # Render targets
            try:
                om = pipe.GetOutputMerger()
                if om:
                    rts = []
                    for i, rt in enumerate(om.renderTargets):
                        if rt.resourceId != rd.ResourceId.Null():
                            rts.append({"index": i, "resource_id": str(rt.resourceId)})
                    pipeline_info["render_targets"] = rts

                    if om.depthTarget.resourceId != rd.ResourceId.Null():
                        pipeline_info["depth_target"] = str(om.depthTarget.resourceId)
            except Exception:
                pass

            # Input assembly
            try:
                ia = pipe.GetIAState()
                if ia:
                    pipeline_info["input_assembly"] = {"topology": str(ia.topology)}
            except Exception:
                pass

            result["pipeline"] = pipeline_info

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["pipeline"]

    def _get_stage_resources(self, controller, pipe, stage, reflection):
        """Get shader resource views (SRVs) for a stage"""
        resources = []
        try:
            srvs = pipe.GetReadOnlyResources(stage, False)

            name_map = {}
            if reflection:
                for res in reflection.readOnlyResources:
                    name_map[res.fixedBindNumber] = res.name

            for srv in srvs:
                if srv.descriptor.resource == rd.ResourceId.Null():
                    continue

                slot = srv.access.index
                res_info = {
                    "slot": slot,
                    "name": name_map.get(slot, ""),
                    "resource_id": str(srv.descriptor.resource),
                }

                res_info.update(
                    self._get_resource_details(controller, srv.descriptor.resource)
                )

                res_info["first_mip"] = srv.descriptor.firstMip
                res_info["num_mips"] = srv.descriptor.numMips
                res_info["first_slice"] = srv.descriptor.firstSlice
                res_info["num_slices"] = srv.descriptor.numSlices

                resources.append(res_info)
        except Exception as e:
            resources.append({"error": str(e)})

        return resources

    def _get_stage_uavs(self, controller, pipe, stage, reflection):
        """Get unordered access views (UAVs) for a stage"""
        uavs = []
        try:
            uav_list = pipe.GetReadWriteResources(stage, False)

            name_map = {}
            if reflection:
                for res in reflection.readWriteResources:
                    name_map[res.fixedBindNumber] = res.name

            for uav in uav_list:
                if uav.descriptor.resource == rd.ResourceId.Null():
                    continue

                slot = uav.access.index
                uav_info = {
                    "slot": slot,
                    "name": name_map.get(slot, ""),
                    "resource_id": str(uav.descriptor.resource),
                }

                uav_info.update(
                    self._get_resource_details(controller, uav.descriptor.resource)
                )

                uav_info["first_element"] = uav.descriptor.firstMip
                uav_info["num_elements"] = uav.descriptor.numMips

                uavs.append(uav_info)
        except Exception as e:
            uavs.append({"error": str(e)})

        return uavs

    def _get_stage_samplers(self, pipe, stage, reflection):
        """Get samplers for a stage"""
        samplers = []
        try:
            sampler_list = pipe.GetSamplers(stage, False)

            name_map = {}
            if reflection:
                for samp in reflection.samplers:
                    name_map[samp.fixedBindNumber] = samp.name

            for samp in sampler_list:
                slot = samp.access.index
                samp_info = {
                    "slot": slot,
                    "name": name_map.get(slot, ""),
                }

                desc = samp.descriptor
                try:
                    samp_info["address_u"] = str(desc.addressU)
                    samp_info["address_v"] = str(desc.addressV)
                    samp_info["address_w"] = str(desc.addressW)
                except AttributeError:
                    pass

                try:
                    samp_info["filter"] = str(desc.filter)
                except AttributeError:
                    pass

                try:
                    samp_info["max_anisotropy"] = desc.maxAnisotropy
                except AttributeError:
                    pass

                try:
                    samp_info["min_lod"] = desc.minLOD
                    samp_info["max_lod"] = desc.maxLOD
                    samp_info["mip_lod_bias"] = desc.mipLODBias
                except AttributeError:
                    pass

                try:
                    samp_info["border_color"] = [
                        desc.borderColor[0],
                        desc.borderColor[1],
                        desc.borderColor[2],
                        desc.borderColor[3],
                    ]
                except (AttributeError, TypeError):
                    pass

                try:
                    samp_info["compare_function"] = str(desc.compareFunction)
                except AttributeError:
                    pass

                samplers.append(samp_info)
        except Exception as e:
            samplers.append({"error": str(e)})

        return samplers

    def _get_stage_cbuffers(self, controller, pipe, stage, reflection):
        """Get constant buffers for a stage from shader reflection"""
        cbuffers = []
        try:
            if not reflection:
                return cbuffers

            for cb in reflection.constantBlocks:
                slot = cb.bindPoint if hasattr(cb, 'bindPoint') else cb.fixedBindNumber
                cb_info = {
                    "slot": slot,
                    "name": cb.name,
                    "byte_size": cb.byteSize,
                    "variable_count": len(cb.variables) if cb.variables else 0,
                    "variables": [],
                }
                if cb.variables:
                    for var in cb.variables:
                        cb_info["variables"].append({
                            "name": var.name,
                            "byte_offset": var.byteOffset,
                            "type": str(var.type.name) if var.type else "",
                        })
                cbuffers.append(cb_info)

        except Exception as e:
            cbuffers.append({"error": str(e)})

        return cbuffers

    def _get_resource_details(self, controller, resource_id):
        """Get details about a resource (texture or buffer)"""
        details = {}

        try:
            resource_name = self.ctx.GetResourceName(resource_id)
            if resource_name:
                details["resource_name"] = resource_name
        except Exception:
            pass

        for tex in controller.GetTextures():
            if tex.resourceId == resource_id:
                details["type"] = "texture"
                details["width"] = tex.width
                details["height"] = tex.height
                details["depth"] = tex.depth
                details["array_size"] = tex.arraysize
                details["mip_levels"] = tex.mips
                details["format"] = str(tex.format.Name())
                details["dimension"] = str(tex.type)
                details["msaa_samples"] = tex.msSamp
                return details

        for buf in controller.GetBuffers():
            if buf.resourceId == resource_id:
                details["type"] = "buffer"
                details["length"] = buf.length
                return details

        return details

    def _get_cbuffer_info(self, controller, pipe, reflection, stage):
        """Get constant buffer information and values"""
        cbuffers = []

        for i, cb in enumerate(reflection.constantBlocks):
            cb_info = {
                "name": cb.name,
                "slot": i,
                "size": cb.byteSize,
                "variables": [],
            }

            try:
                bind = pipe.GetConstantBuffer(stage, i, 0)
                if bind.resourceId != rd.ResourceId.Null():
                    variables = controller.GetCBufferVariableContents(
                        pipe.GetGraphicsPipelineObject(),
                        reflection.resourceId,
                        stage,
                        reflection.entryPoint,
                        i,
                        bind.resourceId,
                        bind.byteOffset,
                        bind.byteSize,
                    )
                    cb_info["variables"] = Serializers.serialize_variables(variables)
            except Exception as e:
                cb_info["error"] = str(e)

            cbuffers.append(cb_info)

        return cbuffers

    def get_cbuffer_contents(self, event_id, stage, slot):
        """Get the contents of a single constant buffer at given event/stage/slot.

        Args:
            event_id: Event ID.
            stage: Shader stage (vertex/pixel/...).
            slot: API real binding point (GL UBO binding / D3D cb register /
                Vulkan binding), as returned by get_pipeline_state cbuffer.slot.
                Use -1 to access non-buffer-backed default block (e.g. GL
                $Globals). The internal reflection index is auto-resolved via
                BindpointMapping.

        Returns: dict {name, slot, reflection_index, size, variables, ...}.
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"cbuffer": None, "error": None}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)

            pipe = controller.GetPipelineState()
            stage_enum = Parsers.parse_stage(stage)

            shader = pipe.GetShader(stage_enum)
            if shader == rd.ResourceId.Null():
                result["error"] = "No %s shader bound" % stage
                return

            reflection = pipe.GetShaderReflection(stage_enum)
            if reflection is None:
                result["error"] = "No reflection for %s shader" % stage
                return

            # Resolve user-provided real-binding `slot` to reflection index via
            # BindpointMapping. slot == -1 means the caller wants the default
            # uniform block / non-buffer-backed cbuffer.
            cb_mapping = None
            try:
                mapping = pipe.GetBindpointMapping(stage_enum)
                if mapping is not None:
                    cb_mapping = mapping.constantBlocks
            except Exception:
                cb_mapping = None

            refl_idx = -1
            matched = []
            for i, cb in enumerate(reflection.constantBlocks):
                bp_idx = cb.bindPoint if hasattr(cb, 'bindPoint') else -1
                real_bind = -1
                if cb_mapping is not None and 0 <= bp_idx < len(cb_mapping):
                    try:
                        real_bind = int(cb_mapping[bp_idx].bind)
                    except Exception:
                        real_bind = -1
                if real_bind == slot:
                    matched.append(i)

            if not matched:
                # Build a hint listing available real bindings for diagnostics.
                avail = []
                for i, cb in enumerate(reflection.constantBlocks):
                    bp_idx = cb.bindPoint if hasattr(cb, 'bindPoint') else -1
                    rb = -1
                    if cb_mapping is not None and 0 <= bp_idx < len(cb_mapping):
                        try:
                            rb = int(cb_mapping[bp_idx].bind)
                        except Exception:
                            rb = -1
                    avail.append("(refl_idx=%d, name=%s, slot=%d)" % (i, cb.name, rb))
                result["error"] = ("No cbuffer with real binding slot=%d in %s shader. "
                                   "Available: %s") % (slot, stage, "; ".join(avail))
                return
            refl_idx = matched[0]

            cb = reflection.constantBlocks[refl_idx]
            bp_idx = cb.bindPoint if hasattr(cb, 'bindPoint') else -1
            cb_info = {
                "name": cb.name,
                "slot": slot,
                "reflection_index": refl_idx,
                "buffer_backed": bool(getattr(cb, 'bufferBacked', True)),
                "size": cb.byteSize,
                "variables": [],
                "bound": False,
            }
            if len(matched) > 1:
                cb_info["ambiguous_matches"] = matched

            try:
                # GetConstantBuffer takes the reflection-level bindPoint index.
                bind = pipe.GetConstantBuffer(stage_enum, bp_idx, 0)
                if bind.resourceId != rd.ResourceId.Null():
                    cb_info["bound"] = True
                    cb_info["resource_id"] = str(bind.resourceId)
                    cb_info["byte_offset"] = bind.byteOffset
                    cb_info["byte_size"] = bind.byteSize
                    variables = controller.GetCBufferVariableContents(
                        pipe.GetGraphicsPipelineObject(),
                        reflection.resourceId,
                        reflection.entryPoint,
                        refl_idx,
                        bind.resourceId,
                        bind.byteOffset,
                        bind.byteSize,
                    )
                    cb_info["variables"] = Serializers.serialize_variables(variables)
                else:
                    cb_info["error"] = ("No constant buffer bound at real binding slot=%d "
                                        "(reflection_index=%d)") % (slot, refl_idx)
            except Exception as e:
                cb_info["error"] = str(e)

            result["cbuffer"] = cb_info

        self._invoke(callback)
        return result

    def _get_resource_bindings(self, reflection):
        """Get shader resource bindings"""
        resources = []

        try:
            for res in reflection.readOnlyResources:
                resources.append(
                    {
                        "name": res.name,
                        "type": str(res.resType),
                        "binding": res.fixedBindNumber,
                        "access": "ReadOnly",
                    }
                )
        except Exception:
            pass

        try:
            for res in reflection.readWriteResources:
                resources.append(
                    {
                        "name": res.name,
                        "type": str(res.resType),
                        "binding": res.fixedBindNumber,
                        "access": "ReadWrite",
                    }
                )
        except Exception:
            pass

        return resources



    def get_postvs(self, event_id, stage="VSOut", instance=0, view=0,
                   first_vertex=0, num_vertices=64, parse_position=True):
        """Get Post Vertex Shader output mesh data for a draw call.

        Args:
            event_id: Event ID of the draw call.
            stage: Mesh stage - "VSIn", "VSOut", or "GSOut" (default "VSOut").
            instance: Instance index for instanced draws (default 0).
            view: View index for multiview rendering (default 0).
            first_vertex: First vertex to read from the output buffer (default 0).
            num_vertices: Max vertices to read, capped at 1024 (default 64).
            parse_position: If True, parse first up-to-4 floats of each vertex
                as a position vector (default True).

        Returns:
            dict with mesh format metadata, raw bytes (base64), and optional
            parsed positions. Returns dict with 'empty': True if the stage has
            no output (e.g. compute-only draw or stage not run).
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        # Cap num_vertices to avoid huge payloads.
        if num_vertices > 1024:
            num_vertices = 1024
        if num_vertices < 0:
            num_vertices = 0

        stage_enum = self._parse_mesh_stage(stage)
        result = {"data": None, "error": None}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)

            mesh_fmt = controller.GetPostVSData(instance, view, stage_enum)

            if mesh_fmt.vertexResourceId == rd.ResourceId.Null():
                result["data"] = {
                    "event_id": event_id,
                    "stage": stage,
                    "instance": instance,
                    "view": view,
                    "empty": True,
                    "reason": "No output for this stage (null vertex resource)",
                }
                return

            # Format info.
            fmt = mesh_fmt.format
            format_info = {
                "comp_type": str(fmt.compType),
                "comp_count": int(fmt.compCount),
                "comp_byte_width": int(fmt.compByteWidth),
            }
            try:
                format_info["name"] = str(fmt.Name())
            except Exception:
                pass

            vertex_stride = int(mesh_fmt.vertexByteStride)
            vertex_offset = int(mesh_fmt.vertexByteOffset)
            num_indices = int(mesh_fmt.numIndices)

            # Compute how many vertices we actually read.
            available = max(0, num_indices - first_vertex)
            read_count = min(num_vertices, available)

            # Cap total bytes to 64KB.
            max_bytes = 64 * 1024
            if vertex_stride > 0 and read_count * vertex_stride > max_bytes:
                read_count = max_bytes // vertex_stride

            read_bytes = read_count * vertex_stride
            read_offset = vertex_offset + first_vertex * vertex_stride

            raw = b""
            if read_bytes > 0:
                try:
                    raw = bytes(controller.GetBufferData(
                        mesh_fmt.vertexResourceId, read_offset, read_bytes))
                except Exception as e:
                    result["error"] = "GetBufferData failed: %s" % str(e)
                    return

            data = {
                "event_id": event_id,
                "stage": stage,
                "instance": instance,
                "view": view,
                "empty": False,
                "topology": str(mesh_fmt.topology),
                "num_indices": num_indices,
                "base_vertex": int(mesh_fmt.baseVertex),
                "near_plane": float(mesh_fmt.nearPlane),
                "far_plane": float(mesh_fmt.farPlane),
                "unproject": bool(mesh_fmt.unproject),
                "flip_y": bool(mesh_fmt.flipY),
                "format": format_info,
                "vertex_resource_id": str(mesh_fmt.vertexResourceId),
                "vertex_byte_offset": vertex_offset,
                "vertex_byte_stride": vertex_stride,
                "index_resource_id": str(mesh_fmt.indexResourceId),
                "index_byte_offset": int(mesh_fmt.indexByteOffset),
                "index_byte_stride": int(mesh_fmt.indexByteStride),
                "first_vertex": first_vertex,
                "vertices_read": read_count,
                "bytes_read": len(raw),
                "content_base64": base64.b64encode(raw).decode("ascii") if raw else "",
            }

            # Parse positions: read up to 4 floats from the start of each vertex.
            if parse_position and read_count > 0 and vertex_stride >= 4:
                positions = []
                comp_count = min(4, format_info["comp_count"] or 4)
                comp_w = format_info["comp_byte_width"] or 4
                # Only attempt float parsing for float-typed formats with 4-byte components.
                # Fall back to interpreting first 16 bytes as 4 floats regardless,
                # which is what RenderDoc's PostVS typically outputs (SV_Position).
                is_float = "Float" in format_info["comp_type"]
                if is_float and comp_w == 4:
                    floats_per_vertex = min(comp_count, vertex_stride // 4)
                    fmt_str = "<%df" % floats_per_vertex
                    fmt_size = floats_per_vertex * 4
                    for i in range(read_count):
                        off = i * vertex_stride
                        if off + fmt_size <= len(raw):
                            try:
                                positions.append(list(struct.unpack(
                                    fmt_str, raw[off:off + fmt_size])))
                            except Exception:
                                break
                else:
                    # Generic fallback: try parsing first 16 bytes as 4 floats.
                    if vertex_stride >= 16:
                        for i in range(read_count):
                            off = i * vertex_stride
                            if off + 16 <= len(raw):
                                try:
                                    positions.append(list(struct.unpack(
                                        "<4f", raw[off:off + 16])))
                                except Exception:
                                    break
                data["parsed_positions"] = positions

            result["data"] = data

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    @staticmethod
    def _parse_mesh_stage(stage_str):
        """Convert mesh stage string to MeshDataStage enum."""
        stage_map = {
            "vsin": rd.MeshDataStage.VSIn,
            "vsout": rd.MeshDataStage.VSOut,
            "gsout": rd.MeshDataStage.GSOut,
        }
        key = stage_str.lower()
        if key not in stage_map:
            raise ValueError("Unknown mesh stage: %s (expected VSIn/VSOut/GSOut)" % stage_str)
        return stage_map[key]

