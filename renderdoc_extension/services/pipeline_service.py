"""
Pipeline state service for RenderDoc.
"""

import renderdoc as rd

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
                render_targets = pipe.GetOutputTargets()
                rts = []
                for i, rt in enumerate(render_targets):
                    if rt.resourceId != rd.ResourceId.Null():
                        rts.append({"index": i, "resource_id": str(rt.resourceId)})
                pipeline_info["render_targets"] = rts

                depth_target = pipe.GetDepthTarget()
                if depth_target.resourceId != rd.ResourceId.Null():
                    pipeline_info["depth_target"] = str(depth_target.resourceId)
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
        """Get shader resource views (SRVs) for a stage (RenderDoc 1.13 API)"""
        resources = []
        try:
            srvs = pipe.GetReadOnlyResources(stage, False)

            # Build (bindset, bind) -> name map via BindpointMapping.
            # In 1.13, ShaderResource.bindPoint is an index into
            # ShaderBindpointMapping.readOnlyResources, which holds the real Bindpoint.
            name_by_bp = {}
            try:
                mapping = pipe.GetBindpointMapping(stage)
                if reflection and mapping:
                    ro_map = mapping.readOnlyResources
                    for i, res in enumerate(reflection.readOnlyResources):
                        if i < len(ro_map):
                            bp = ro_map[i]
                            name_by_bp[(bp.bindset, bp.bind)] = res.name
            except Exception:
                pass

            for arr in srvs:
                bp = arr.bindPoint
                key = (bp.bindset, bp.bind)
                name = name_by_bp.get(key, "")
                for res in arr.resources:
                    if res.resourceId == rd.ResourceId.Null():
                        continue
                    res_info = {
                        "bindset": bp.bindset,
                        "slot": bp.bind,
                        "name": name,
                        "resource_id": str(res.resourceId),
                        "first_mip": res.firstMip,
                        "first_slice": res.firstSlice,
                    }
                    res_info.update(
                        self._get_resource_details(controller, res.resourceId)
                    )
                    resources.append(res_info)
        except Exception as e:
            resources.append({"error": str(e)})

        return resources

    def _get_stage_uavs(self, controller, pipe, stage, reflection):
        """Get unordered access views (UAVs) for a stage (RenderDoc 1.13 API)"""
        uavs = []
        try:
            uav_list = pipe.GetReadWriteResources(stage, False)

            name_by_bp = {}
            try:
                mapping = pipe.GetBindpointMapping(stage)
                if reflection and mapping:
                    rw_map = mapping.readWriteResources
                    for i, res in enumerate(reflection.readWriteResources):
                        if i < len(rw_map):
                            bp = rw_map[i]
                            name_by_bp[(bp.bindset, bp.bind)] = res.name
            except Exception:
                pass

            for arr in uav_list:
                bp = arr.bindPoint
                key = (bp.bindset, bp.bind)
                name = name_by_bp.get(key, "")
                for res in arr.resources:
                    if res.resourceId == rd.ResourceId.Null():
                        continue
                    uav_info = {
                        "bindset": bp.bindset,
                        "slot": bp.bind,
                        "name": name,
                        "resource_id": str(res.resourceId),
                        "first_mip": res.firstMip,
                        "first_slice": res.firstSlice,
                    }
                    uav_info.update(
                        self._get_resource_details(controller, res.resourceId)
                    )
                    uavs.append(uav_info)
        except Exception as e:
            uavs.append({"error": str(e)})

        return uavs

    def _get_stage_samplers(self, pipe, stage, reflection):
        """Get samplers for a stage (RenderDoc 1.13 API)

        Note: 1.13's GetSamplers takes only the stage argument, and the returned
        BoundResourceArray.resources items are BoundResource entries that only
        carry resourceId (no descriptor with addressU/filter/etc.). For full
        sampler descriptor access on 1.13 the API-specific pipeline state
        (D3D11Pipe / VKPipe / GLPipe) is required, which is intentionally not
        attempted here to keep this generic.
        """
        samplers = []
        try:
            sampler_list = pipe.GetSamplers(stage)

            name_by_bp = {}
            try:
                mapping = pipe.GetBindpointMapping(stage)
                if reflection and mapping:
                    samp_map = mapping.samplers
                    for i, samp in enumerate(reflection.samplers):
                        if i < len(samp_map):
                            bp = samp_map[i]
                            name_by_bp[(bp.bindset, bp.bind)] = samp.name
            except Exception:
                pass

            for arr in sampler_list:
                bp = arr.bindPoint
                key = (bp.bindset, bp.bind)
                name = name_by_bp.get(key, "")
                for res in arr.resources:
                    samplers.append({
                        "bindset": bp.bindset,
                        "slot": bp.bind,
                        "name": name,
                        "sampler_id": str(res.resourceId),
                    })
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
                # 1.13: ConstantBlock.bindPoint is int32 (mapping index)
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
                        # 1.13: ShaderConstantType has no 'name'; use type.descriptor.name
                        type_name = ""
                        try:
                            type_name = var.type.descriptor.name
                        except AttributeError:
                            try:
                                type_name = var.type.name
                            except AttributeError:
                                pass
                        cb_info["variables"].append({
                            "name": var.name,
                            "byte_offset": var.byteOffset,
                            "type": str(type_name) if type_name else "",
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

    def _get_resource_bindings(self, reflection):
        """Get shader resource bindings"""
        resources = []

        try:
            for res in reflection.readOnlyResources:
                resources.append(
                    {
                        "name": res.name,
                        "type": str(res.resType),
                        "binding": res.bindPoint,
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
                        "binding": res.bindPoint,
                        "access": "ReadWrite",
                    }
                )
        except Exception:
            pass

        return resources
