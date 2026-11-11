"""Experimental Seer-v2 subclass with one high-resolution temporal fusion stage.

The archived v2 implementation and evaluator are untouched. Existing checkpoint
keys are unchanged; every new trainable state is under temporal_fusion.*.
"""
from __future__ import annotations

import torch

from .local_illumination_cues import extract_illumination_cues
from .pixel_branch_v2 import SeerV2PixelBranch
from .temporal_detail_fusion import TemporalDetailFusion


class SeerFusionPixelBranch(SeerV2PixelBranch):
    def __init__(self, *, fusion_mode="prior", fusion_scale=3, fusion_radius=2,
                 fusion_offset=2, fusion_tap=2, fusion_logit_scale=8.0, **kwargs):
        if fusion_mode not in ("local", "prior"):
            raise ValueError("fusion_mode must be local or prior")
        if fusion_scale not in (1, 2, 3):
            raise ValueError("fusion_scale must be 1, 2 or 3")
        if not 0 <= fusion_tap < 4:
            raise ValueError("fusion_tap must be a zero-based index in [0, 3]")
        super().__init__(**kwargs)
        self.fusion_mode, self.fusion_scale, self.fusion_tap = fusion_mode, fusion_scale, fusion_tap
        self._fusion_config = dict(fusion_mode=fusion_mode, fusion_scale=fusion_scale,
                                   fusion_radius=fusion_radius, fusion_offset=fusion_offset,
                                   fusion_tap=fusion_tap, fusion_logit_scale=fusion_logit_scale)
        self.temporal_fusion = TemporalDetailFusion(
            self.dims[fusion_scale], radius=fusion_radius, offset=fusion_offset,
            guidance=float(fusion_mode == "prior"), token_grid=self.vjepa_grid,
            logit_scale=fusion_logit_scale)

    def get_fusion_config(self):
        """Constructor kwargs to save alongside base architecture/checkpoint args."""
        return dict(self._fusion_config)

    def forward(self, video_lq, vjepa_layers, teacher_layers=None, *, fusion_bypass=False, return_aux=False):
        # This follows the frozen v2 forward verbatim except for the one marked
        # encoder-stage insertion and the additional scalar diagnostics.
        _, _, time, _, _ = video_lq.shape
        x_bt = self.stem(self._bt(video_lq))
        cues = extract_illumination_cues(video_lq) if self.use_illumination_cues else None
        skips, fusion_aux = [], {}
        for scale in range(5):
            for block in self.enc_blocks[scale]:
                x_bt = (torch.utils.checkpoint.checkpoint(block, x_bt, use_reentrant=False)
                        if self.use_checkpoint and self.training else block(x_bt))
            if self.enc_xattn[scale] is not None:
                x_bt = self.enc_xattn[scale](x_bt, self._kv(vjepa_layers, scale, time))
            if self.enc_illum[scale] is not None and cues is not None:
                x_bt = self.enc_illum[scale](x_bt, cues, target_T=time)
            if scale == self.fusion_scale:
                tokens = vjepa_layers[self.fusion_tap] if self.fusion_mode == "prior" else None
                teacher = teacher_layers[self.fusion_tap] if teacher_layers is not None else None
                volume, fusion_aux = self.temporal_fusion(
                    self._back(x_bt, time), tokens, teacher,
                    bypass=fusion_bypass, return_aux=return_aux)
                x_bt = self._bt(volume)
            if scale < 4:
                skips.append(x_bt)
                x_bt = self.enc_down[scale](x_bt)
            else:
                volume = self.temporal_mixer(self._back(x_bt, time))
                if self.cfim_bottleneck is not None:
                    volume = self.cfim_bottleneck(volume)
                x_bt = self._bt(volume)
        for index, scale in enumerate([3, 2, 1, 0]):
            x_bt = self.dec_up[index](x_bt)
            x_bt = self.dec_reduce[index](torch.cat([x_bt, skips[scale]], dim=1))
            for block in self.dec_blocks[index]:
                x_bt = (torch.utils.checkpoint.checkpoint(block, x_bt, use_reentrant=False)
                        if self.use_checkpoint and self.training else block(x_bt))
            if self.dec_xattn[index] is not None:
                x_bt = self.dec_xattn[index](x_bt, self._kv(vjepa_layers, scale, time))
            if self.dec_illum[index] is not None and cues is not None:
                x_bt = self.dec_illum[index](x_bt, cues, target_T=time)
        residual = self._back(self.residual_head(x_bt), time)
        reflectance = self._back(self.retinex_head_R(x_bt), time)
        illumination = self._back(self.retinex_head_L(x_bt), time)
        output = video_lq + residual
        if not self.training:
            output, reflectance, illumination = (v.clamp(0, 1) for v in (output, reflectance, illumination))
        return {"pred_pixels": output, "residual": residual, "R_hat": reflectance,
                "L_hat": illumination, "fusion_aux": fusion_aux}
