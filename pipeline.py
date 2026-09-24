import math
from dataclasses import dataclass

from PIL import Image
from tinygrad import Device, Tensor, TinyJit, dtypes, nn
from tinygrad.nn import RMSNorm
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


class TimestepEmbedder:
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        self.frequency_embedding_size = frequency_embedding_size
        self.half = frequency_embedding_size // 2
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size),
            None,
            nn.Linear(hidden_size, hidden_size),
        ]
        self.freqs = 10000.0 ** (
            -Tensor.arange(self.half, dtype=dtypes.float32) / self.half
        ).is_param_(False)

    def __call__(self, t: Tensor) -> Tensor:
        freqs = self.freqs.to(t.device)
        args = t.cast(dtypes.float32).reshape(-1, 1) * freqs.reshape(1, self.half)
        emb = Tensor.cat(args.cos(), args.sin(), dim=-1)
        emb = emb.cast(self.mlp[0].weight.dtype)
        x = self.mlp[0](emb)
        x = x.silu()
        return self.mlp[2](x)


class BottleneckPatchEmbed:
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_channels=3,
        pca_channels=128,
        hidden_size=1248,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj1 = nn.Conv2d(
            in_channels,
            pca_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(
            pca_channels, hidden_size, kernel_size=1, stride=1, bias=True
        )

    def __call__(self, x):
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)


class SwiGLUMlp:
    def __init__(self, in_features: int, hidden_features: int):
        hidden_dim = (hidden_features + 7) // 8 * 8
        self.w1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w3 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_features, bias=False)

    def __call__(self, x):
        return self.w2(self.w1(x).silu() * self.w3(x))


class TextRotaryEmbedding1D:
    def __init__(self, head_dim: int, theta: float = 10000.0):
        self.head_dim = head_dim
        self.theta = theta

    def __call__(self, x):
        b, length, h, d = x.shape
        inv = 1.0 / (self.theta ** (Tensor.arange(0, d, 2, dtype=dtypes.float32) / d))
        pos = Tensor.arange(length, dtype=dtypes.float32)
        angles = pos.reshape(-1, 1) * inv.reshape(1, -1)
        cos = angles.cos().cast(x.dtype).reshape(1, length, 1, -1)
        sin = angles.sin().cast(x.dtype).reshape(1, length, 1, -1)
        x1, x2 = x.chunk(2, dim=-1)
        return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)


class VisionRotaryEmbeddingFast:
    def __init__(self, head_dim: int, theta: float = 10000.0):
        self.dim = head_dim // 2
        self.theta = theta

    def __call__(self, x):
        length = x.shape[1]
        side = int(math.sqrt(length))
        if side * side != length:
            raise ValueError(f"image token length must be square, got {length}")
        freqs = 1.0 / (
            self.theta
            ** (Tensor.arange(0, self.dim, 2, dtype=dtypes.float32) / self.dim)
        )
        t = Tensor.arange(side, dtype=dtypes.float32)
        base = t.reshape(-1, 1) * freqs.reshape(1, -1)
        f_h = base.reshape(side, 1, -1).expand(side, side, -1)
        f_w = base.reshape(1, side, -1).expand(side, side, -1)
        angles = f_h.cat(f_w, dim=-1)
        angles = angles.reshape(length, -1)
        cos = angles.cos().cast(x.dtype).reshape(1, length, 1, -1)
        sin = angles.sin().cast(x.dtype).reshape(1, length, 1, -1)
        x1, x2 = x.chunk(2, dim=-1)
        return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)


class MultiModalRotaryEmbeddingFast:
    def __init__(self, head_dim: int):
        self.text_rope = TextRotaryEmbedding1D(head_dim)
        self.vision_rope = VisionRotaryEmbeddingFast(head_dim)

    def __call__(self, x, txt_len: int):
        txt = self.text_rope(x[:, :txt_len])
        img = self.vision_rope(x[:, txt_len:])
        return txt.cat(img, dim=1)


class PlainTextTransformerBlock:
    def __init__(self, hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.attn_proj = nn.Linear(inner_dim, hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = TextRotaryEmbedding1D(head_dim)

    def __call__(self, txt):
        b, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(
            b, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = Tensor.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b, length, -1)
        txt = txt + self.attn_proj(out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class DoubleStreamDiTBlock:
    def __init__(
        self,
        hidden_size=1248,
        txt_hidden_size=1248,
        num_heads=24,
        head_dim=52,
        mlp_ratio=2.7,
    ):
        self.hidden_size = hidden_size
        self.txt_hidden_size = txt_hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(txt_hidden_size)
        self.txt_norm2 = RMSNorm(txt_hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.txt_qkv = nn.Linear(txt_hidden_size, inner_dim * 3)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = MultiModalRotaryEmbeddingFast(head_dim)
        self.img_attn_proj = nn.Linear(inner_dim, hidden_size)
        self.txt_attn_proj = nn.Linear(inner_dim, txt_hidden_size)
        self.img_mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLUMlp(txt_hidden_size, int(txt_hidden_size * mlp_ratio))

    def __call__(self, x, txt, vec):
        b, li, _ = x.shape
        lt = txt.shape[1]
        x_norm = self.img_norm1(x)
        txt_norm = self.txt_norm1(txt)
        qkv_i = self.img_qkv(x_norm).reshape(b, li, 3, self.num_heads, self.head_dim)
        qkv_t = self.txt_qkv(txt_norm).reshape(b, lt, 3, self.num_heads, self.head_dim)
        q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
        q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        q = self.rope(q_t.cat(q_i, dim=1), txt_len=lt)
        k = self.rope(k_t.cat(k_i, dim=1), txt_len=lt)
        v = v_t.cat(v_i, dim=1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = Tensor.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3)
        x = x + self.img_attn_proj(out[:, lt:].reshape(b, li, -1))
        txt = txt + self.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


class FinalLayer:
    def __init__(self, hidden_size=1248, patch_size=16, out_channels=3):
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def __call__(self, x, vec=None):
        return self.linear(self.norm_final(x))


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = Tensor.arange(grid_size, dtype=dtypes.float32)
    grid_w = Tensor.arange(grid_size, dtype=dtypes.float32)
    grid = Tensor.meshgrid(grid_w, grid_h, indexing="xy")
    grid = Tensor.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return Tensor.cat(emb_h, emb_w, dim=1)


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = Tensor.arange(embed_dim // 2, dtype=dtypes.float32).to(pos.device)
    omega = 1.0 / (10000.0 ** (omega / (embed_dim / 2.0)))
    out = pos.reshape(-1, 1) * omega
    return Tensor.cat(out.sin(), out.cos(), dim=1)


@dataclass
class MMJiTConfig:
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 3
    txt_input_size: int = 1024
    hidden_size: int = 768
    txt_hidden_size: int = 768
    cond_vec_size: int = 768
    depth_double: int = 17
    txt_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128
    prompt_length: int = 256
    n_T: int = 100
    prediction: str = "x"
    sampler: str = "euler"
    cfg_channels: int = 3
    cfg_interval: tuple = (0.0, 1.0)
    llm: str = "google/flan-t5-large"


class MMJiT:
    def __init__(self, cfg: MMJiTConfig):
        self.cfg = cfg
        self.latent_img_size = cfg.image_size // cfg.patch_size
        self.img_embedder = BottleneckPatchEmbed(
            cfg.image_size,
            cfg.patch_size,
            cfg.in_channels,
            cfg.pca_channels,
            cfg.hidden_size,
        )
        self.txt_embedder = nn.Linear(
            cfg.txt_input_size, cfg.txt_hidden_size, bias=False
        )
        self.mask_token = Tensor.zeros(1, 1, cfg.txt_input_size)
        self.t_embedder = TimestepEmbedder(cfg.cond_vec_size)
        self.pooled_embedder = nn.Linear(
            cfg.txt_input_size, cfg.cond_vec_size, bias=False
        )
        self.txt_preamble_blocks = [
            PlainTextTransformerBlock(
                cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio
            )
            for _ in range(cfg.txt_preamble_depth)
        ]
        self.double_blocks = [
            DoubleStreamDiTBlock(
                cfg.hidden_size,
                cfg.txt_hidden_size,
                cfg.num_heads,
                cfg.head_dim,
                cfg.mlp_ratio,
            )
            for _ in range(cfg.depth_double)
        ]
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)
        self.pos = get_2d_sincos_pos_embed(
            self.cfg.hidden_size, self.latent_img_size
        ).is_param_(False)

    def unpatchify(self, x):
        b = x.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, h * p, w * p)

    def __call__(self, img, t, context, attn_mask):
        if img.ndim == 4 and img.shape[1] != self.cfg.in_channels:
            img = img.permute(0, 3, 1, 2)
        attn_mask = attn_mask.to(context.device)
        context = Tensor.where(
            attn_mask[:, :, None] > 0.5, context, self.mask_token.cast(context.dtype)
        )
        x = self.img_embedder(img)
        x = x + self.pos.cast(x.dtype).to(x.device)[None]
        t_vec = self.t_embedder(t)
        txt = self.txt_embedder(context.cast(self.txt_embedder.weight.dtype))
        pooled_text = context.mean(axis=1)
        vec = t_vec + self.pooled_embedder(
            pooled_text.cast(self.pooled_embedder.weight.dtype)
        )
        for block in self.txt_preamble_blocks:
            txt = block(txt)
        for block in self.double_blocks:
            x, txt = block(x, txt, vec)
        combined = Tensor.cat(txt, x, dim=1)
        out = self.final_layer(combined, vec)
        img_out = out[:, txt.shape[1] :, :]
        return self.unpatchify(img_out)


class DiffusionModel:
    def __init__(self, cfg: MMJiTConfig | None = None):
        self.cfg = cfg or MMJiTConfig()
        self.net = MMJiT(self.cfg)

    def real_t_to_embed_t(self, t):
        return t

    def pred_velocity(self, x, t, text, mask):
        x0 = self.net(x, self.real_t_to_embed_t(t), text, mask)
        return (x0 - x) / Tensor.clamp(1 - t[:, None, None, None], min_=0.001)

    def cfg_velocity(self, x, t, text, mask, cfg_scale: float):
        b = x.shape[0]
        xx = x.cat(x, dim=0)
        tt = t.cat(t, dim=0)
        yy = text.cat(text, dim=0)
        mm = mask.cat(Tensor.zeros_like(mask), dim=0)
        out = self.pred_velocity(xx, tt, yy, mm)
        cond, uncond = out[:b], out[b:]
        use_cfg = (
            (t >= self.cfg.cfg_interval[0]) & (t <= self.cfg.cfg_interval[1])
        ).cast(out.dtype)
        scale = use_cfg.reshape(-1, 1, 1, 1) * (cfg_scale - 1.0) + 1.0
        return uncond + (cond - uncond) * scale

    def sample(self, text, mask, cfg_scale=6.0, progress=False):
        b = text.shape[0]
        device = text.device
        dtype = text.dtype
        x = (
            Tensor.randn(
                b,
                self.cfg.in_channels,
                self.cfg.image_size,
                self.cfg.image_size,
                dtype=dtype,
                device=device,
            )
            * 2.0
        )
        x = x.realize()
        text = text.realize()
        mask = mask.realize()
        timesteps = (
            Tensor.linspace(0.0, 1.0, self.cfg.n_T + 1, dtype=dtype)
            .to(device)
            .realize()
        )
        t_cur_in = Tensor.empty(b, dtype=dtype, device=device)
        t_next_in = Tensor.empty(b, dtype=dtype, device=device)
        iterator = range(self.cfg.n_T)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator)

        @TinyJit
        def _jit_step(x, t_cur, t_next, text, mask):
            v = self.cfg_velocity(x, t_cur, text, mask, cfg_scale)
            x = x + (t_next - t_cur)[:, None, None, None] * v
            return x.realize()

        for i in iterator:
            t_cur_in.assign(timesteps[i].expand(b)).realize()
            t_next_in.assign(timesteps[i + 1].expand(b)).realize()
            x = _jit_step(x, t_cur_in, t_next_in, text, mask)
        return x


class MiniT2IMMJiTModel:
    def __init__(self, cfg=None):
        self.cfg = cfg or MMJiTConfig()
        self.model = DiffusionModel(self.cfg)


class MiniT2ITextToImagePipeline:
    def __init__(self, transformer: MiniT2IMMJiTModel):
        text_encoder_name = MMJiTConfig.llm
        self.transformer = transformer
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_name,
            torch_dtype="float32",
        )

    def _encode_prompt(self, prompt: str | list[str]):
        cfg = self.transformer.cfg
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.prompt_length,
        )
        input_ids = tokens.input_ids.to(self.text_encoder.device)
        attn = tokens.attention_mask.to(self.text_encoder.device)
        text = self.text_encoder(
            input_ids=input_ids, attention_mask=attn
        ).last_hidden_state
        return (
            Tensor(text.detach().cpu().numpy(), device=Device.DEFAULT),
            Tensor(attn.detach().cpu().numpy(), device=Device.DEFAULT),
        )

    def __call__(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: int | None = None,
        output_type: str = "pil",
        progress: bool = True,
    ):
        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for p in prompt:
                prompt_batch.extend([p] * num_images_per_prompt)

        old_steps = self.transformer.cfg.n_T
        self.transformer.model.cfg.n_T = int(num_inference_steps or old_steps)
        try:
            text, attn = self._encode_prompt(prompt_batch)
            model_dtype = dtypes.bfloat16
            images = self.transformer.model.sample(
                text.cast(dtype=model_dtype),
                attn.cast(dtype=model_dtype),
                cfg_scale=guidance_scale,
                progress=progress,
            )
        finally:
            self.transformer.model.cfg.n_T = old_steps

        images = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).cast(dtypes.uint8)
        images = images.permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            images = [Image.fromarray(image) for image in images]
        return images
