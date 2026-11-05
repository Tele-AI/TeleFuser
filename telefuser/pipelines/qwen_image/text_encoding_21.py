"""Qwen-Image 2.1 prompt encoding stage."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager


class TextEncoding21Stage(BaseStage):
    """Encode prompts with the ModuleManager-owned Qwen3-VL model."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig):
        super().__init__(name, model_runtime_config)
        self.text_encoder = module_manager.fetch_module("text_encoder")
        self.processor = module_manager.fetch_module("processor")
        self.model_names = ["text_encoder"]
        self.system_prompt = "Comprehend and analyze the provided prompt."
        self.prompt_template = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        sys_message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        self.drop_idx = len(self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
        self.image_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        bool_mask = mask.bool()
        lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, lengths.tolist(), dim=0))

    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def process(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        images: list[Image.Image] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return positive/negative embeddings, masks, and image-slot masks."""

        positive = self._encode(prompt, images)
        negative = self._encode(negative_prompt, images) if negative_prompt is not None else None
        pos_embeds, pos_mask, pos_image_mask = positive
        neg_embeds = neg_mask = neg_image_mask = None
        if negative is not None:
            neg_embeds, neg_mask, neg_image_mask = negative
        batch_size, seq_len, _ = pos_embeds.shape
        pos_embeds = pos_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        pos_mask = pos_mask.repeat_interleave(num_images_per_prompt, dim=0) if pos_mask is not None else None
        pos_image_mask = pos_image_mask.repeat_interleave(num_images_per_prompt, dim=0)
        if neg_embeds is not None:
            neg_embeds = neg_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            neg_mask = neg_mask.repeat_interleave(num_images_per_prompt, dim=0) if neg_mask is not None else None
            neg_image_mask = neg_image_mask.repeat_interleave(num_images_per_prompt, dim=0)
        del batch_size, seq_len
        return pos_embeds, pos_mask, pos_image_mask, neg_embeds, neg_mask, neg_image_mask

    def _encode(
        self, prompt: str | list[str], images: list[Image.Image] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if images:
            image_prefix = " ".join(
                f"<image{index}><|vision_start|><|image_pad|><|vision_end|>" for index in range(1, len(images) + 1)
            )
            prompts = [self.prompt_template.format(f"{image_prefix}{text or ' '}") for text in prompts]
            vision_images = []
            for _ in prompts:
                for image in images:
                    if image.mode == "RGBA":
                        white = Image.new("RGB", image.size, (255, 255, 255))
                        white.paste(image, mask=image.getchannel("A"))
                        vision_images.append(white)
                    else:
                        vision_images.append(image)
        else:
            prompts = [self.prompt_template.format(text or " ") for text in prompts]
            vision_images = None
        processor_kwargs = {
            "text": prompts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if vision_images is not None:
            processor_kwargs["images"] = vision_images
        inputs = self.processor(**processor_kwargs).to(self.device)
        model = getattr(self.text_encoder, "model", self.text_encoder)
        language_model = getattr(model, "language_model", model)
        hook = language_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            forward_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "output_hidden_states": True,
            }
            if vision_images is not None and hasattr(inputs, "pixel_values"):
                forward_kwargs["pixel_values"] = inputs.pixel_values
                forward_kwargs["image_grid_thw"] = inputs.image_grid_thw
            if hasattr(inputs, "mm_token_type_ids"):
                forward_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            hook.remove()
        hidden = outputs.hidden_states[-1]
        split_hidden = [item[self.drop_idx :] for item in self._extract_masked_hidden(hidden, inputs.attention_mask)]
        image_masks = [
            (ids[mask.bool()] == self.image_token_id)[self.drop_idx :]
            for ids, mask in zip(inputs.input_ids, inputs.attention_mask)
        ]
        max_len = max(item.shape[0] for item in split_hidden)
        embeds = torch.stack(
            [torch.cat([item, item.new_zeros(max_len - item.shape[0], item.shape[1])]) for item in split_hidden]
        )
        valid_masks = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(item.shape[0], dtype=torch.bool, device=item.device),
                        torch.zeros(max_len - item.shape[0], dtype=torch.bool, device=item.device),
                    ]
                )
                for item in split_hidden
            ]
        )
        image_mask = torch.stack(
            [
                torch.cat([item, torch.zeros(max_len - item.shape[0], dtype=torch.bool, device=item.device)])
                for item in image_masks
            ]
        )
        if bool(valid_masks.all()):
            valid_masks = None
        return embeds.to(dtype=self.torch_dtype), valid_masks, image_mask
