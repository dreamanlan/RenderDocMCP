"""
Resource information service for RenderDoc.
"""

import base64

import renderdoc as rd

from ..utils import Parsers


class ResourceService:
    """Resource information service"""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def _find_texture_by_id(self, controller, resource_id):
        """Find texture by resource ID"""
        target_id = Parsers.extract_numeric_id(resource_id)
        for tex in controller.GetTextures():
            tex_id_str = str(tex.resourceId)
            tex_id = Parsers.extract_numeric_id(tex_id_str)
            if tex_id == target_id:
                return tex
        return None

    def get_buffer_contents(self, resource_id, offset=0, length=0):
        """Get buffer data"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None, "error": None}

        def callback(controller):
            # Parse resource ID
            try:
                rid = Parsers.parse_resource_id(resource_id)
            except Exception:
                result["error"] = "Invalid resource ID: %s" % resource_id
                return

            # Find buffer
            buf_desc = None
            for buf in controller.GetBuffers():
                if buf.resourceId == rid:
                    buf_desc = buf
                    break

            if not buf_desc:
                result["error"] = "Buffer not found: %s" % resource_id
                return

            # Get data
            actual_length = length if length > 0 else buf_desc.length
            data = controller.GetBufferData(rid, offset, actual_length)

            result["data"] = {
                "resource_id": resource_id,
                "length": len(data),
                "total_size": buf_desc.length,
                "offset": offset,
                "content_base64": base64.b64encode(data).decode("ascii"),
            }

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    def get_texture_info(self, resource_id):
        """Get texture metadata"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"texture": None, "error": None}

        def callback(controller):
            try:
                tex_desc = self._find_texture_by_id(controller, resource_id)

                if not tex_desc:
                    result["error"] = "Texture not found: %s" % resource_id
                    return

                result["texture"] = {
                    "resource_id": resource_id,
                    "width": tex_desc.width,
                    "height": tex_desc.height,
                    "depth": tex_desc.depth,
                    "array_size": tex_desc.arraysize,
                    "mip_levels": tex_desc.mips,
                    "format": str(tex_desc.format.Name()),
                    "dimension": str(tex_desc.type),
                    "msaa_samples": tex_desc.msSamp,
                    "byte_size": tex_desc.byteSize,
                }
            except Exception as e:
                import traceback
                result["error"] = "Error: %s\n%s" % (str(e), traceback.format_exc())

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["texture"]

    def get_texture_data(self, resource_id, mip=0, slice=0, sample=0, depth_slice=None, event_id=None, x=None, y=None, w=None, h=None):
        """Get texture pixel data at a specific event (default: last event in frame)."""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None, "error": None}

        def callback(controller):
            # Switch to specific event if requested (critical for RT pool reuse cases)
            if event_id is not None:
                controller.SetFrameEvent(event_id, False)
            tex_desc = self._find_texture_by_id(controller, resource_id)

            if not tex_desc:
                result["error"] = "Texture not found: %s" % resource_id
                return

            # Validate mip level
            if mip < 0 or mip >= tex_desc.mips:
                result["error"] = "Invalid mip level %d (texture has %d mips)" % (
                    mip,
                    tex_desc.mips,
                )
                return

            # Validate slice for array/cube textures
            max_slices = tex_desc.arraysize
            if tex_desc.cubemap:
                max_slices = tex_desc.arraysize * 6
            if slice < 0 or (max_slices > 1 and slice >= max_slices):
                result["error"] = "Invalid slice %d (texture has %d slices)" % (
                    slice,
                    max_slices,
                )
                return

            # Validate sample for MSAA
            if sample < 0 or (tex_desc.msSamp > 1 and sample >= tex_desc.msSamp):
                result["error"] = "Invalid sample %d (texture has %d samples)" % (
                    sample,
                    tex_desc.msSamp,
                )
                return

            # Calculate dimensions at this mip level
            mip_width = max(1, tex_desc.width >> mip)
            mip_height = max(1, tex_desc.height >> mip)
            mip_depth = max(1, tex_desc.depth >> mip)

            # Validate depth_slice for 3D textures
            is_3d = tex_desc.depth > 1
            if depth_slice is not None:
                if not is_3d:
                    result["error"] = "depth_slice can only be used with 3D textures"
                    return
                if depth_slice < 0 or depth_slice >= mip_depth:
                    result["error"] = "Invalid depth_slice %d (texture has %d depth at mip %d)" % (
                        depth_slice,
                        mip_depth,
                        mip,
                    )
                    return

            # Validate crop region: must be all-None or all-set
            crop_params = [x, y, w, h]
            crop_set_count = sum(1 for p in crop_params if p is not None)
            do_crop = (crop_set_count == 4)
            if crop_set_count not in (0, 4):
                result["error"] = "Crop region requires all of x/y/w/h or none"
                return
            if do_crop:
                if w <= 0 or h <= 0:
                    result["error"] = "Crop w/h must be positive"
                    return
                if x < 0 or y < 0:
                    result["error"] = "Crop x/y must be non-negative"
                    return
                if x + w > mip_width or y + h > mip_height:
                    result["error"] = "Crop region (%d,%d,%dx%d) exceeds mip size %dx%d" % (
                        x, y, w, h, mip_width, mip_height,
                    )
                    return
                if is_3d and depth_slice is None:
                    result["error"] = "3D texture crop requires depth_slice to be specified"
                    return

            # Create subresource specification
            sub = rd.Subresource()
            sub.mip = mip
            sub.slice = slice
            sub.sample = sample

            # Get texture data
            try:
                data = controller.GetTextureData(tex_desc.resourceId, sub)
            except Exception as e:
                result["error"] = "Failed to get texture data: %s" % str(e)
                return

            # Extract depth slice for 3D textures if requested
            output_depth = mip_depth
            if is_3d and depth_slice is not None:
                total_size = len(data)
                bytes_per_slice = total_size // mip_depth
                slice_start = depth_slice * bytes_per_slice
                slice_end = slice_start + bytes_per_slice
                data = data[slice_start:slice_end]
                output_depth = 1

            # Apply 2D crop region (per row slicing)
            out_width = mip_width
            out_height = mip_height
            if do_crop:
                # Reject compressed formats for crop
                if tex_desc.format.type != rd.ResourceFormatType.Regular:
                    result["error"] = "Crop region not supported for compressed format: %s" % str(tex_desc.format.Name())
                    return
                bpp = tex_desc.format.compCount * tex_desc.format.compByteWidth
                if bpp <= 0:
                    result["error"] = "Cannot determine bytes_per_pixel for format: %s" % str(tex_desc.format.Name())
                    return
                bytes_per_row = mip_width * bpp
                expected_len = bytes_per_row * mip_height
                if len(data) < expected_len:
                    result["error"] = "Data length %d shorter than expected %d for crop" % (len(data), expected_len)
                    return
                cropped = bytearray()
                row_byte_w = w * bpp
                for row in range(h):
                    src_y = y + row
                    src_off = src_y * bytes_per_row + x * bpp
                    cropped.extend(data[src_off:src_off + row_byte_w])
                data = bytes(cropped)
                out_width = w
                out_height = h

            result["data"] = {
                "resource_id": resource_id,
                "width": out_width,
                "height": out_height,
                "depth": output_depth,
                "mip": mip,
                "slice": slice,
                "sample": sample,
                "depth_slice": depth_slice,
                "format": str(tex_desc.format.Name()),
                "dimension": str(tex_desc.type),
                "is_3d": is_3d,
                "total_depth": mip_depth if is_3d else 1,
                "mip_width": mip_width,
                "mip_height": mip_height,
                "crop_x": x if do_crop else None,
                "crop_y": y if do_crop else None,
                "crop_w": w if do_crop else None,
                "crop_h": h if do_crop else None,
                "data_length": len(data),
                "content_base64": base64.b64encode(data).decode("ascii"),
            }

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    def pick_pixel(self, resource_id, x, y, mip=0, slice=0, sample=0, event_id=None, type_cast="typeless"):
        """Read a single pixel value from a texture.

        Returns 4 components in float/uint/sint form. Bypasses bulk data transfer:
        ideal for diagnosing per-pixel issues (e.g. black-screen analysis) when
        the full texture would be too large to ship through the MCP channel.
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        # Map type_cast string to rd.CompType
        type_map = {
            "typeless": rd.CompType.Typeless,
            "float": rd.CompType.Float,
            "unorm": rd.CompType.UNorm,
            "snorm": rd.CompType.SNorm,
            "uint": rd.CompType.UInt,
            "sint": rd.CompType.SInt,
            "depth": rd.CompType.Depth,
            "unorm_srgb": rd.CompType.UNormSRGB,
        }
        cast_key = (type_cast or "typeless").lower()
        if cast_key not in type_map:
            raise ValueError("Invalid type_cast '%s'. Valid: %s" % (type_cast, ", ".join(type_map.keys())))
        comp_type = type_map[cast_key]

        result = {"data": None, "error": None}

        def callback(controller):
            # Switch to specific event if requested (critical for RT pool reuse cases)
            if event_id is not None:
                controller.SetFrameEvent(event_id, False)

            tex_desc = self._find_texture_by_id(controller, resource_id)
            if tex_desc is None:
                result["error"] = "Texture not found: %s" % resource_id
                return

            # Validate coordinates against mip-level dimensions
            mip_width = max(1, tex_desc.width >> mip)
            mip_height = max(1, tex_desc.height >> mip)
            if x < 0 or x >= mip_width or y < 0 or y >= mip_height:
                result["error"] = "Pixel (%d,%d) out of range for mip %d (%dx%d)" % (x, y, mip, mip_width, mip_height)
                return

            # Create subresource specification
            sub = rd.Subresource()
            sub.mip = mip
            sub.slice = slice
            sub.sample = sample

            try:
                pixel_value = controller.PickPixel(tex_desc.resourceId, x, y, sub, comp_type)
            except Exception as e:
                result["error"] = "PickPixel failed: %s" % str(e)
                return

            # PixelValue has floatValue/uintValue/intValue (each 4-component array)
            result["data"] = {
                "resource_id": resource_id,
                "x": int(x),
                "y": int(y),
                "mip": int(mip),
                "slice": int(slice),
                "sample": int(sample),
                "event_id": event_id,
                "type_cast": cast_key,
                "format": str(tex_desc.format.Name()),
                "mip_width": int(mip_width),
                "mip_height": int(mip_height),
                "float_value": [float(v) for v in pixel_value.floatValue],
                "uint_value": [int(v) for v in pixel_value.uintValue],
                "int_value": [int(v) for v in pixel_value.intValue],
            }

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    def save_texture(self, resource_id, file_path, file_format, mip, slice, sample, event_id, type_cast, alpha_handling):
        """
        Save a texture to disk as image file (PNG/JPG/BMP/TGA/HDR/EXR/DDS).

        Useful for visual inspection of large render targets that exceed
        MCP transport size limits. The file is written by the RenderDoc
        replay process directly to the host filesystem.
        """
        result = {"data": None, "error": None}

        def callback(controller):
            try:
                if event_id is not None:
                    controller.SetFrameEvent(int(event_id), False)

                tex_desc = self._find_texture_by_id(controller, resource_id)
                if tex_desc is None:
                    result["error"] = "Texture not found: %s" % resource_id
                    return

                save_data = rd.TextureSave()
                save_data.resourceId = tex_desc.resourceId

                # File type
                fmt = file_format.upper()
                fmt_map = {
                    "PNG": rd.FileType.PNG,
                    "JPG": rd.FileType.JPG,
                    "JPEG": rd.FileType.JPG,
                    "BMP": rd.FileType.BMP,
                    "TGA": rd.FileType.TGA,
                    "HDR": rd.FileType.HDR,
                    "EXR": rd.FileType.EXR,
                    "DDS": rd.FileType.DDS,
                }
                if fmt not in fmt_map:
                    result["error"] = "Unsupported file_format: " + file_format
                    return
                save_data.destType = fmt_map[fmt]

                # Subresource selection
                save_data.mip = int(mip)
                save_data.slice.sliceIndex = int(slice)
                save_data.slice.cubeCruciform = False
                save_data.sample.sampleIndex = int(sample)
                save_data.sample.mapToArray = False

                # Type cast
                cast_map = {
                    "typeless": rd.CompType.Typeless,
                    "float": rd.CompType.Float,
                    "unorm": rd.CompType.UNorm,
                    "snorm": rd.CompType.SNorm,
                    "uint": rd.CompType.UInt,
                    "sint": rd.CompType.SInt,
                    "depth": rd.CompType.Depth,
                    "unorm_srgb": rd.CompType.UNormSRGB,
                }
                save_data.typeCast = cast_map.get(type_cast.lower(), rd.CompType.Typeless)

                # Alpha handling
                alpha_map = {
                    "discard": rd.AlphaMapping.Discard,
                    "preserve": rd.AlphaMapping.Preserve,
                    "blend_to_color": rd.AlphaMapping.BlendToColor,
                    "blend_to_checkerboard": rd.AlphaMapping.BlendToCheckerboard,
                }
                save_data.alpha = alpha_map.get(alpha_handling.lower(), rd.AlphaMapping.Discard)

                # JPEG quality default
                save_data.jpegQuality = 90

                # Single channel and range
                save_data.channelExtract = -1

                ok = controller.SaveTexture(save_data, file_path)
                # ok may be ResultDetails or bool depending on binding version
                success = bool(ok) if not hasattr(ok, "OK") else ok.OK()
                err_msg = "" if success else (ok.Message() if hasattr(ok, "Message") else "SaveTexture failed")

                result["data"] = {
                    "success": success,
                    "file_path": file_path,
                    "file_format": fmt,
                    "resource_id": str(tex_desc.resourceId),
                    "event_id": event_id,
                    "error_message": err_msg,
                }
            except Exception as e:
                result["error"] = str(e)

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

