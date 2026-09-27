"""V-JEPA-LLVE model package.

v6 (current main path):
    build_vjepa_backbone        → frozen V-JEPA 2.1 ViT-L feature extractor
    VJepaLatentDiT              → DiT velocity field in V-JEPA latent space
    VJepaPixelDecoder           → 4-scale UNet, V-JEPA cross-attn at every scale
    VJepaLLVE_v6                → end-to-end model

Engineering trick (NOT in v6 paper main path; ablation only):
    CounterfactualRefiner       → V-JEPA Predictor body + LoRA + identity-gate

Legacy components from earlier iterations (kept for backward-compat reading
of older configs / ckpts; new training should NOT use these):
    SDVAEWrapper                → SD VAE (v5 used; v6 dropped)
    DiTVelocityField            → DiT in SD VAE latent (v5; v6 uses VJepaLatentDiT)
    VJepaDiffusion              → v5 model (SD VAE + V-JEPA + DiT + refiner)
"""
from .vjepa_backbone import VJepaBackboneLocal21, build_vjepa_backbone
from .latent_dit import VJepaLatentDiT
from .pixel_decoder import VJepaPixelDecoder
from .vjepa_llve_v6 import VJepaLLVE_v6

# Optional engineering trick (used only as ablation)
from .predictor_lora import CounterfactualRefiner, LoRALinear

# Legacy v5 (not used by v6 main path)
from .vae_wrapper import SDVAEWrapper
from .dit_velocity import DiTVelocityField
from .vjepa_diffusion import VJepaDiffusion

__all__ = [
    "VJepaBackboneLocal21", "build_vjepa_backbone",
    "VJepaLatentDiT", "VJepaPixelDecoder", "VJepaLLVE_v6",
    "CounterfactualRefiner", "LoRALinear",
    "SDVAEWrapper", "DiTVelocityField", "VJepaDiffusion",
]
