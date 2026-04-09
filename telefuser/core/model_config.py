"""Model detection and loading configurations.

Defines model loader configurations for automatic model type detection
from state dict hashes. Each entry maps state dict signatures to
model names, classes, and source formats.
"""

from __future__ import annotations

from ..models.TCDecoder import TAEHV
from ..models.flashvsr_dit import FlashVSRModel
from ..models.flux2_dit import Flux2DiT
from ..models.hunyuan_video_dit import HunyuanVideoDiT
from ..models.liveact_dit import LiveActDiT
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel
from ..models.ltx_dit import LTXVideoTransformer
from ..models.ltx_gemma_text_encoder import GemmaTextEncoder, LTXEmbeddingsProcessor
from ..models.ltx_upsampler import LTXSpatialUpsampler
from ..models.ltx_video_vae import LTXVideoVAE
from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..models.realesrgan import RealESRGAN
from ..models.rift_hdv3 import IFNet
from ..models.wan22_video_vae import Wan22VideoVAE
from ..models.wan_video_dit import WanModel
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.z_image_dit import ZImageTransformer2DModel
from ..models.z_image_text_encoder import ZImageTextEncoder

# Model loader configurations for automatic type detection.
# Format: (keys_hash_without_shape, keys_hash_with_shape, model_names, model_classes, model_resource)
# Hash is MD5 of sorted state dict keys (with optional shape suffixes).
model_loader_configs = [
    # LTX 2.3 dev shared checkpoint
    (
        None,
        "f3a83ecf3995dcc4fae2d27e08ad5767",
        ["ltx_embeddings_processor", "ltx_video_vae", "ltx_dit"],
        [LTXEmbeddingsProcessor, LTXVideoVAE, LTXVideoTransformer],
        "official",
    ),
    # LTX 2.3 spatial upsampler
    (None, "aed408774d694a2452f69936c32febb5", ["ltx_spatial_upsampler"], [LTXSpatialUpsampler], "official"),
    # LTX 2.3 Gemma Text Encoder
    (None, "33917f31c4a79196171154cca39f165e", ["gemma_text_encoder"], [GemmaTextEncoder], "official"),
    # WanVideo VAE
    (None, "4c3523c69fb7b24cf2db147a715b277f", ["wan_video_decoder"], [TAEHV], "official"),
    # QwenImage VAE and Text Encoder
    (None, "ed4ea5824d55ec3107b09815e318123a", ["qwen_image_vae"], [QwenImageVAE], "official"),
    (None, "8004730443f55db63092006dd9f7110e", ["qwen_image_text_encoder"], [QwenImageTextEncoder], "official"),
    # QwenImage DiT (various formats)
    (None, "7a32c4aa3de140d48a5899ca505944b9", ["qwen_image_dit"], [QwenImageDiT], "official"),  # fp8 scaled
    (None, "0319a1cb19835fb510907dd3367c95ff", ["qwen_image_dit"], [QwenImageDiT], "official"),
    # WanVideo DiT - Wan2.1 1.3B variants
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "official"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "official"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "official"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "official"),
    (None, "b3aba5f6fddb5e117640e751591db89f", ["wan_video_dit"], [WanModel], "official"),
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "official"),
    # WanVideo DiT - diffusers format
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "7cf3a086b49216bded0728ce78d59687", ["wan_video_dit"], [WanModel], "diffusers"),
    # WanVideo DiT - Wan2.2 A14B variants
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "official"),  # I2V
    (None, "4cf556355bc7e9b6545b38f4930f60b1", ["wan_video_dit"], [WanModel], "official"),  # I2V high noise fp8
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "official"),  # high/low noise
    (None, "9d0240d8e7650a9ec65b2b617cc9c357", ["wan_video_dit"], [WanModel], "official"),  # A14B T2V high noise fp8
    # WanVideo DiT - Wan2.2 5B TI2V (unified T2V/I2V with blended latent)
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "official"),  # 5B TI2V
    # WanVideo Text Encoder and Image Encoder
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "official"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "official"),
    # WanVideo VAE variants
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "official"),
    (None, "19560d299104e665df05de9a03074ed5", ["wan_video_vae"], [WanVideoVAE], "official"),  # light 75% pruning
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "official"),
    # light decoder + original encoder
    (None, "e9addbd0c9d54bc1827116b98e0dd1a0", ["wan_video_vae"], [WanVideoVAE], "official"),
    # Wan2.2 VAE (48 latent channels)
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [Wan22VideoVAE], "official"),
    # FlashVSR
    (None, "0f889085aa6209c79f284d963d6cbe95", ["flashvsr_dit"], [FlashVSRModel], "official"),
    # HunyuanVideo DiT
    (None, "91d733b509142990a1a1bfa1516a0de0", ["hunyuan_video_dit"], [HunyuanVideoDiT], "official"),
    # HunyuanVideo DiT - diffusers format
    (None, "b2c3d4e5f678901234567890123456a7", ["hunyuan_video_dit"], [HunyuanVideoDiT], "diffusers"),
    # Z-Image
    (None, "0f050f62a88876fea6eae0a18dac5a2e", ["zimage_text_encoder"], [ZImageTextEncoder], "diffusers"),
    (None, "fc3a8a1247fe185ce116ccbe0e426c28", ["zimage_dit"], [ZImageTransformer2DModel], "diffusers"),  # turbo dit
    # Flux2-klein-9b distill
    (None, "39c6fc48f07bebecedbbaa971ff466c8", ["transformer"], [Flux2DiT], "diffusers"),
    # VFI (Video Frame Interpolation)
    (None, "47ac734bfc54bc2f75ff43e8e016588d", ["vfi_model"], [IFNet], "official"),
    # ISR (Image Super Resolution)
    (None, "d93074599b84fa126cbc6fb8bf61ea6e", ["upscaler_model"], [RealESRGAN], "official"),
    # LongCat Video DiT
    (
        None,
        "8b27900f680d7251ce44e2dc8ae1ffef",
        ["wan_video_dit"],
        [LongCatVideoTransformer3DModel],
        "official",
    ),
    # LiveAct DiT (Audio-conditioned I2V)
    (None, "99c7bbedc6dd6c4824799e4f999fee0a", ["liveact_dit"], [LiveActDiT], "official"),
]
