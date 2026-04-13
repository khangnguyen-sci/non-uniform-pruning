import os
import torch
import types
from torch import nn

from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

from forward import (
    MyUNet2DConditionModel_SD_forward,
    MyCrossAttnDownBlock2D_SD_forward,
    MyDownBlock2D_SD_forward,
    MyUNetMidBlock2DCrossAttn_SD_forward,
    MyCrossAttnUpBlock2D_SD_forward,
    MyUpBlock2D_SD_forward,
    MyResnetBlock2D_SD_forward,
    MyTransformer2DModel_SD_forward,
)

_SD_BASE = [320, 640, 1280, 1280]
_BASELINE_R = 0.75
_BASELINE_C = [int(c * _BASELINE_R) for c in _SD_BASE]  


def _get_ratio(name: str, default: float = 0.75) -> float:
    return float(os.environ.get(name, default))


def find_parent(model, module_name: str):
    comps = module_name.split(".")
    parent = model
    for c in comps[:-1]:
        parent = getattr(parent, c)
    return parent, comps[-1]


def _pick_group_count(num_channels: int) -> int:
    for g in [32, 24, 16, 12, 8, 6, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1


def _slice_copy_param(dst: torch.Tensor, src: torch.Tensor):
    slices = tuple(slice(0, s) for s in dst.shape)
    dst.copy_(src[slices])


def _prune_conv(conv: nn.Conv2d, in_c: int, out_c: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        _slice_copy_param(new_conv.weight, conv.weight)
        if conv.bias is not None:
            _slice_copy_param(new_conv.bias, conv.bias)
    return new_conv


def _prune_linear(lin: nn.Linear, in_f: int, out_f: int) -> nn.Linear:
    new_lin = nn.Linear(in_features=in_f, out_features=out_f, bias=(lin.bias is not None))
    with torch.no_grad():
        _slice_copy_param(new_lin.weight, lin.weight)
        if lin.bias is not None:
            _slice_copy_param(new_lin.bias, lin.bias)
    return new_lin


def _prune_groupnorm(gn: nn.GroupNorm, num_ch: int) -> nn.GroupNorm:
    g = _pick_group_count(num_ch)
    new_gn = nn.GroupNorm(num_groups=g, num_channels=num_ch, eps=gn.eps, affine=gn.affine)
    with torch.no_grad():
        if gn.affine:
            _slice_copy_param(new_gn.weight, gn.weight)
            _slice_copy_param(new_gn.bias, gn.bias)
    return new_gn


def _prune_layernorm(ln: nn.LayerNorm, n: int) -> nn.LayerNorm:
    new_ln = nn.LayerNorm(n, eps=ln.eps, elementwise_affine=ln.elementwise_affine)
    with torch.no_grad():
        if ln.elementwise_affine:
            _slice_copy_param(new_ln.weight, ln.weight)
            _slice_copy_param(new_ln.bias, ln.bias)
    return new_ln


def _prune_resnet_block(res: ResnetBlock2D, in_c: int, out_c: int):
    # norm1 is on x_in (in_c)
    if hasattr(res, "norm1") and isinstance(res.norm1, nn.GroupNorm):
        res.norm1 = _prune_groupnorm(res.norm1, in_c)

    # conv1: in_c -> out_c
    if hasattr(res, "conv1") and isinstance(res.conv1, nn.Conv2d):
        res.conv1 = _prune_conv(res.conv1, in_c, out_c)

    # norm2 is on hidden (out_c)
    if hasattr(res, "norm2") and isinstance(res.norm2, nn.GroupNorm):
        res.norm2 = _prune_groupnorm(res.norm2, out_c)

    # conv2: out_c -> out_c
    if hasattr(res, "conv2") and isinstance(res.conv2, nn.Conv2d):
        res.conv2 = _prune_conv(res.conv2, out_c, out_c)

    # shortcut
    if in_c != out_c:
        if hasattr(res, "conv_shortcut") and isinstance(res.conv_shortcut, nn.Conv2d):
            res.conv_shortcut = _prune_conv(res.conv_shortcut, in_c, out_c)
        else:
            res.conv_shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)

    res.in_channels = in_c
    res.out_channels = out_c


def _prune_transformer2d(t2d: Transformer2DModel, c_new: int):
    # norm
    if hasattr(t2d, "norm") and isinstance(t2d.norm, nn.GroupNorm):
        t2d.norm = _prune_groupnorm(t2d.norm, c_new)

    # proj_in/proj_out
    if hasattr(t2d, "proj_in"):
        if isinstance(t2d.proj_in, nn.Linear):
            t2d.proj_in = _prune_linear(t2d.proj_in, c_new, c_new)
        elif isinstance(t2d.proj_in, nn.Conv2d):
            t2d.proj_in = _prune_conv(t2d.proj_in, c_new, c_new)

    if hasattr(t2d, "proj_out"):
        if isinstance(t2d.proj_out, nn.Linear):
            t2d.proj_out = _prune_linear(t2d.proj_out, c_new, c_new)
        elif isinstance(t2d.proj_out, nn.Conv2d):
            t2d.proj_out = _prune_conv(t2d.proj_out, c_new, c_new)

    # inner blocks
    for blk in getattr(t2d, "transformer_blocks", []):
        # norms (LayerNorm in SD)
        if hasattr(blk, "norm1") and isinstance(blk.norm1, nn.LayerNorm):
            blk.norm1 = _prune_layernorm(blk.norm1, c_new)
        if hasattr(blk, "norm3") and isinstance(blk.norm3, nn.LayerNorm):
            blk.norm3 = _prune_layernorm(blk.norm3, c_new)

        # attn1 linears
        if hasattr(blk, "attn1"):
            attn = blk.attn1

            for attr in ["to_q", "to_k", "to_v"]:
                if hasattr(attn, attr) and isinstance(getattr(attn, attr), nn.Linear):
                    setattr(attn, attr, _prune_linear(getattr(attn, attr), c_new, c_new))

            if hasattr(attn, "to_out"):
                to_out = attn.to_out

                if isinstance(to_out, (nn.ModuleList, nn.Sequential)) and len(to_out) > 0:
                    if isinstance(to_out[0], nn.Linear):
                        to_out[0] = _prune_linear(to_out[0], c_new, c_new)
                elif isinstance(to_out, nn.Linear):
                    attn.to_out = _prune_linear(to_out, c_new, c_new)

        if hasattr(blk, "ff") and hasattr(blk.ff, "net") and isinstance(blk.ff.net, nn.ModuleList):
            if hasattr(blk.ff.net[0], "proj") and isinstance(blk.ff.net[0].proj, nn.Linear):
                old = blk.ff.net[0].proj
                old_in = old.in_features
                old_out = old.out_features
                mult0 = max(1, int(round(old_out / float(old_in))))
                blk.ff.net[0].proj = _prune_linear(old, c_new, mult0 * c_new)

            if len(blk.ff.net) > 2 and isinstance(blk.ff.net[2], nn.Linear):
                old = blk.ff.net[2]
                old_in = old.in_features
                old_out = old.out_features
                mult2 = max(1, int(round(old_in / float(old_out))))
                blk.ff.net[2] = _prune_linear(old, mult2 * c_new, c_new)


def _channel_plan_from_env():
    r0 = _get_ratio("ADCSR_R0", 0.75)
    r1 = _get_ratio("ADCSR_R1", 0.75)
    r2 = _get_ratio("ADCSR_R2", 0.75)
    r3 = _get_ratio("ADCSR_R3", 0.75)
    rmid = _get_ratio("ADCSR_RMID", r3)

    if abs(r0 - 0.75) > 1e-6:
        raise ValueError(
            "For this AdcSR setup (decoder unchanged), ADCSR_R0 must be 0.75. "
            "Changing it breaks conv_out->decoder.mid_block interface."
        )

    c0 = int(_SD_BASE[0] * 0.75) 
    c1 = int(_SD_BASE[1] * r1)
    c2 = int(_SD_BASE[2] * r2)
    c3 = int(_SD_BASE[3] * r3)
    cm = int(_SD_BASE[3] * rmid)

    if c1 > _BASELINE_C[1] or c2 > _BASELINE_C[2] or c3 > _BASELINE_C[3]:
        raise ValueError("Stage ratios must be <= 0.75 (so target channels do not exceed params200 architecture).")

    if cm != c3:
        raise ValueError(
            f"Pure pruning requires ADCSR_RMID == ADCSR_R3"
            f"Got CMID={cm}, C3={c3}."
        )

    return c0, c1, c2, c3


def _apply_stage_pruning(unet: nn.Module):
    C0, C1, C2, C3 = _channel_plan_from_env()

    if [C0, C1, C2, C3] == _BASELINE_C:
        return

    # DOWN blocks
    for res in unet.down_blocks[0].resnets:
        _prune_resnet_block(res, C0, C0)

    _prune_resnet_block(unet.down_blocks[1].resnets[0], C0, C1)
    _prune_resnet_block(unet.down_blocks[1].resnets[1], C1, C1)

    _prune_resnet_block(unet.down_blocks[2].resnets[0], C1, C2)
    _prune_resnet_block(unet.down_blocks[2].resnets[1], C2, C2)

    _prune_resnet_block(unet.down_blocks[3].resnets[0], C2, C3)
    _prune_resnet_block(unet.down_blocks[3].resnets[1], C3, C3)

    # MID
    for res in unet.mid_block.resnets:
        _prune_resnet_block(res, C3, C3)

    skip_stack = [C0]
    skip_stack += [C0, C0] + [C0]
    skip_stack += [C1, C1] + [C1]
    skip_stack += [C2, C2] + [C2]
    skip_stack += [C3, C3]

    def pop_skip():
        if not skip_stack:
            raise RuntimeError("Internal error: skip_stack underflow while planning up-block pruning.")
        return skip_stack.pop()

    hidden_c = C3

    def prune_up_block(up_blk, out_c):
        nonlocal hidden_c
        for i in range(3):
            skip_c = pop_skip()
            in_c = hidden_c + skip_c
            _prune_resnet_block(up_blk.resnets[i], in_c, out_c)
            hidden_c = out_c

        if hasattr(up_blk, "upsamplers") and up_blk.upsamplers is not None:
            for up in up_blk.upsamplers:
                if isinstance(up, Upsample2D) and hasattr(up, "conv") and isinstance(up.conv, nn.Conv2d):
                    up.conv = _prune_conv(up.conv, out_c, out_c)
                    up.channels = out_c

    prune_up_block(unet.up_blocks[0], C3)
    prune_up_block(unet.up_blocks[1], C2)
    prune_up_block(unet.up_blocks[2], C1)
    prune_up_block(unet.up_blocks[3], C0)

    if len(skip_stack) != 0:
        raise RuntimeError(f"Internal error: skip_stack not empty after pruning plan ({len(skip_stack)} left).")

    def stage_channel_for_name(name: str) -> int:
        if name.startswith("down_blocks.0"):
            return C0
        if name.startswith("down_blocks.1"):
            return C1
        if name.startswith("down_blocks.2"):
            return C2
        if name.startswith("down_blocks.3"):
            return C3
        if name.startswith("mid_block"):
            return C3
        if name.startswith("up_blocks.0"):
            return C3
        if name.startswith("up_blocks.1"):
            return C2
        if name.startswith("up_blocks.2"):
            return C1
        if name.startswith("up_blocks.3"):
            return C0
        if name.startswith("conv_norm_out") or name.startswith("conv_act") or name.startswith("conv_out"):
            return C0
        return C0

    if isinstance(unet.conv_in, nn.Conv2d):
        if unet.conv_in.out_channels != C0:
            unet.conv_in = _prune_conv(unet.conv_in, unet.conv_in.in_channels, C0)

    if hasattr(unet, "conv_norm_out") and isinstance(unet.conv_norm_out, nn.GroupNorm):
        if unet.conv_norm_out.num_channels != C0:
            unet.conv_norm_out = _prune_groupnorm(unet.conv_norm_out, C0)

    if isinstance(unet.conv_out, nn.Conv2d):
        if unet.conv_out.in_channels != C0 or unet.conv_out.out_channels != 256:
            new_conv_out = nn.Conv2d(
                in_channels=C0,
                out_channels=256,
                kernel_size=unet.conv_out.kernel_size,
                stride=unet.conv_out.stride,
                padding=unet.conv_out.padding,
                dilation=unet.conv_out.dilation,
                groups=unet.conv_out.groups,
                bias=(unet.conv_out.bias is not None),
            )
            with torch.no_grad():
                _slice_copy_param(new_conv_out.weight, unet.conv_out.weight)
                if unet.conv_out.bias is not None:
                    _slice_copy_param(new_conv_out.bias, unet.conv_out.bias)
            unet.conv_out = new_conv_out

    for name, module in unet.named_modules():
        if isinstance(module, Transformer2DModel):
            c = stage_channel_for_name(name)
            _prune_transformer2d(module, c)
        elif isinstance(module, Downsample2D):
            c = stage_channel_for_name(name)
            if hasattr(module, "conv") and isinstance(module.conv, nn.Conv2d):
                if module.conv.in_channels != c or module.conv.out_channels != c:
                    module.conv = _prune_conv(module.conv, c, c)
            module.channels = c


def halve_channels(model):
    for name, module in model.named_modules():
        if hasattr(module, "pruned"):
            continue

        if isinstance(module, nn.Conv2d):
            in_channels = int(module.in_channels * 0.75)
            out_channels = int(module.out_channels * 0.75)
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(module.weight[:out_channels, :in_channels])
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias[:out_channels])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_conv)
            new_conv.pruned = True

        elif isinstance(module, nn.Linear):
            in_features = int(module.in_features * 0.75)
            out_features = int(module.out_features * 0.75)
            new_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=module.bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(module.weight[:out_features, :in_features])
                if module.bias is not None:
                    new_linear.bias.copy_(module.bias[:out_features])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_linear)
            new_linear.pruned = True

        elif isinstance(module, nn.GroupNorm):
            num_channels = int(module.num_channels * 0.75)
            for num_groups in [32, 24, 16, 12, 8, 6, 4, 2, 1]:
                if num_channels % num_groups == 0:
                    break
            new_gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=module.eps, affine=module.affine)
            with torch.no_grad():
                new_gn.weight.copy_(module.weight[:num_channels])
                new_gn.bias.copy_(module.bias[:num_channels])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_gn)
            new_gn.pruned = True

        elif isinstance(module, nn.LayerNorm):
            normalized_shape = int(module.normalized_shape[0] * 0.75)
            new_ln = nn.LayerNorm(normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine)
            with torch.no_grad():
                new_ln.weight.copy_(module.weight[:normalized_shape])
                new_ln.bias.copy_(module.bias[:normalized_shape])
            parent, last_name = find_parent(model, name)
            setattr(parent, last_name, new_ln)
            new_ln.pruned = True

        elif isinstance(module, (Downsample2D, Upsample2D)):
            module.channels = int(module.channels * 0.75)


class Net(nn.Module):
    def __init__(self, unet, decoder):
        super().__init__()

        del unet.time_embedding

        new_conv_in = nn.Conv2d(16, 320, 3, padding=1)
        new_conv_in.weight.data = unet.conv_in.weight.data.repeat(1, 4, 1, 1)
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in

        new_conv_out = nn.Conv2d(320, 342, 3, padding=1)
        new_conv_out.weight.data = unet.conv_out.weight.data.repeat(86, 1, 1, 1)[:342]
        new_conv_out.bias.data = unet.conv_out.bias.data.repeat(86,)[:342]
        unet.conv_out = new_conv_out

        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj

        unet.apply(ResnetBlock2D_remove_time_emb_proj)

        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                del module.attn2, module.norm2

        unet.apply(BasicTransformerBlock_remove_cross_attn)

        def set_inplace_to_true(module):
            if isinstance(module, (nn.Dropout, nn.SiLU)):
                module.inplace = True

        unet.apply(set_inplace_to_true)

        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_forward, module)
            elif isinstance(module, DownBlock2D):
                module.forward = types.MethodType(MyDownBlock2D_SD_forward, module)
            elif isinstance(module, UNetMidBlock2DCrossAttn):
                module.forward = types.MethodType(MyUNetMidBlock2DCrossAttn_SD_forward, module)
            elif isinstance(module, UpBlock2D):
                module.forward = types.MethodType(MyUpBlock2D_SD_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_forward, module)

        unet.apply(replace_forward_methods)
        unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward, unet)

        # 1) baseline uniform prune
        halve_channels(unet)

        # 2) stage-wise extra pruning (ratios < 0.75)
        _apply_stage_pruning(unet)

        unet.body = nn.Sequential(
            *unet.down_blocks,
            unet.mid_block,
            *unet.up_blocks,
            unet.conv_norm_out,
            unet.conv_act,
            unet.conv_out,
        )

        del decoder.conv_in, decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out

        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            unet,
            decoder.mid_block,
        )

    def forward(self, x):
        return self.body(x)
