## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


from multiprocessing import context
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, center_crop, normalize
from transformers import CLIPModel, CLIPImageProcessor

#### ----------------Original Cross Attention & Guidance & Refinement---------------  #####
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.scale = (inner_dim // num_heads) ** -0.5
        
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, context):
        query_norm = self.norm_query(query)
        context_norm = self.norm_context(context)
        
        q = self.to_q(query_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        attention_update = self.to_out(output)

        return attention_update
    
class ImageGuidanceAttention(nn.Module):
    def __init__(self, 
                 image_dim,
                 text_dim,
                 inner_dim,
                 num_heads=8, 
                 patch_size=4,
                 dropout=0.0):
        super().__init__()
        
        self.patch_size = patch_size
        self.image_dim = image_dim
        
        patch_dim = self.image_dim * (self.patch_size ** 2)
        self.patch_dim = patch_dim

        self.pos_embed = None
        
        self.attention = CrossAttention(
            query_dim=patch_dim,
            context_dim=text_dim,
            inner_dim=inner_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.alpha = nn.Parameter(torch.tensor(1e-4))

    def forward(self, image_feat, text_feat):
        B, C, H, W = image_feat.shape
        p = self.patch_size
        H_new, W_new = H // p, W // p
        
        raw_patches = rearrange(image_feat, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)

        if self.pos_embed is None or self.pos_embed.shape[1] != raw_patches.shape[1]:
            self.pos_embed = nn.Parameter(torch.randn(1, H_new * W_new, self.patch_dim)).to(raw_patches.device)
        
        query_with_pos = raw_patches + self.pos_embed
        attended_patches = self.attention(query=query_with_pos, context=text_feat)
        
        reconstructed_update = rearrange(attended_patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', 
                                         h=H_new, w=W_new, c=C, p1=p, p2=p)
        
        return image_feat + self.alpha * reconstructed_update

class TextRefinementAttention(nn.Module):
    def __init__(self, 
                 text_dim, 
                 image_dim,
                 inner_dim,
                 num_heads=8,
                 patch_size=4,
                 dropout=0.0):
        super().__init__()

        self.patch_size = patch_size
        self.image_dim = image_dim
        
        patch_dim = self.image_dim * (self.patch_size ** 2)
        self.patch_dim = patch_dim

        self.pos_embed = None
        
        self.attention = CrossAttention(
            query_dim=text_dim,
            context_dim=patch_dim,
            inner_dim=inner_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.alpha = nn.Parameter(torch.tensor(1e-4))
        

    def forward(self, text_feat, image_feat):
        B, C, H, W = image_feat.shape
        p = self.patch_size
        H_new, W_new = H // p, W // p

        raw_patches = rearrange(image_feat, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        
        if self.pos_embed is None or self.pos_embed.shape[1] != raw_patches.shape[1]:
            self.pos_embed = nn.Parameter(torch.randn(1, H_new * W_new, self.patch_dim)).to(raw_patches.device)
            
        context_with_pos = raw_patches + self.pos_embed
        text_update = self.attention(query=text_feat, context=context_with_pos)

        return text_feat + self.alpha * text_update


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 is_guidable=False,
                 context_dim=None, 
                 cross_attn_inner_dim=None, 
                 cross_attn_heads=None, 
                 patch_size=4):
        super(TransformerBlock, self).__init__()

        self.is_guidable = is_guidable

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        if self.is_guidable:
            assert all(p is not None for p in [context_dim, cross_attn_inner_dim, cross_attn_heads]), \
                "For a guidable block, context_dim, cross_attn_inner_dim, and cross_attn_heads must be provided."
            
            self.guidance = ImageGuidanceAttention(
                image_dim=dim,
                text_dim=context_dim,
                inner_dim=cross_attn_inner_dim,
                num_heads=cross_attn_heads,
                patch_size=patch_size
            )
        else:
            self.guidance = None

    def forward(self, x, context=None):
        if self.is_guidable and context is not None:
            x_conditioned = self.guidance(x, context)
        else:
            x_conditioned = x
    
        x_out = x_conditioned + self.attn(self.norm1(x_conditioned))
        x_out = x_out + self.ffn(self.norm2(x_out))

        return x_out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
# original
class SESymUnet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, 
                 enc_blk_nums=[], dec_blk_nums=[], 
                 clip_model_name="openai/clip-vit-large-patch14",
                 cross_attn_inner_dims=[48, 96, 192],
                 cross_attn_heads=[1, 2, 4],
                 middle_cross_attn_inner_dim=384,
                 middle_cross_attn_heads=8,
                 restormer_heads=[1, 2, 4],
                 restormer_middle_heads=8,
                 semantic_start_iter=0,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias', 
                 patch_sizes=[8, 8, 8], 
                 middle_patch_size=8,):
        super().__init__()
        
        self.current_iter = 0
        self.semantic_start_iter = semantic_start_iter
        self.training_phase = 1 if self.semantic_start_iter > 0 else 2
        
        num_stages = len(enc_blk_nums)
        assert len(cross_attn_inner_dims) == num_stages, "Length of cross_attn_inner_dims must match the number of encoder/decoder stages."
        assert len(cross_attn_heads) == num_stages, "Length of cross_attn_heads must match the number of encoder/decoder stages."
        assert len(restormer_heads) == num_stages, "Length of restormer_heads must match the number of encoder/decoder stages."

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print(f"Loading FULL CLIP Model: {clip_model_name}")
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_resize_size = self.clip_image_processor.size["shortest_edge"]
        self.clip_crop_size = self.clip_image_processor.crop_size["height"]
        self.clip_normalize_mean = self.clip_image_processor.image_mean
        self.clip_normalize_std = self.clip_image_processor.image_std
        semantic_dim = self.clip_model.config.vision_config.hidden_size
        
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoder_guide_attentions = nn.ModuleList()
        self.decoder_refine_attentions = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_heads[i], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2
            
        self.middle_blks = nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_middle_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(middle_blk_num)])
        
        self.middle_guide_attention = ImageGuidanceAttention(
            image_dim=chan, 
            text_dim=semantic_dim, 
            inner_dim=middle_cross_attn_inner_dim,
            num_heads=middle_cross_attn_heads,
            patch_size=middle_patch_size
        )
        self.middle_refine_attention = TextRefinementAttention(
            text_dim=semantic_dim, 
            image_dim=chan, 
            inner_dim=middle_cross_attn_inner_dim,
            num_heads=middle_cross_attn_heads,
            patch_size=middle_patch_size
        )

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            
            stage_idx = num_stages - 1 - i
            p_decoder = patch_sizes[stage_idx]
            current_cross_attn_inner_dim = cross_attn_inner_dims[stage_idx]
            current_cross_attn_heads = cross_attn_heads[stage_idx]
            current_restormer_heads = restormer_heads[stage_idx]
            
            self.decoder_guide_attentions.append(
                ImageGuidanceAttention(
                    image_dim=chan,
                    text_dim=semantic_dim,
                    inner_dim=current_cross_attn_inner_dim,
                    num_heads=current_cross_attn_heads,
                    patch_size=p_decoder  # 传入当前阶段的patch_size
                )
            )
            self.decoder_refine_attentions.append(
                TextRefinementAttention(
                    text_dim=semantic_dim,
                    image_dim=chan,
                    inner_dim=current_cross_attn_inner_dim,
                    num_heads=current_cross_attn_heads,
                    patch_size=p_decoder
                )
            )
            self.decoders.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=current_restormer_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            
        self.padder_size = (2 ** len(self.encoders)) * 4
        
        if self.semantic_start_iter > 0:
            print(f"--- [RestormerText] Two-stage training enabled. ---")
            print(f"--- Phase 1 (Backbone Training) will run for {self.semantic_start_iter} iterations. ---")
            self.freeze_semantic_modules()
        else:
            pass
    
    def _get_clip_features(self, img):
        b, c, h, w = img.shape
        if h < w:
            new_h = self.clip_resize_size
            new_w = int(w * (self.clip_resize_size / h))
        else:
            new_h = int(h * (self.clip_resize_size / w))
            new_w = self.clip_resize_size
            
        img_resized = resize(
            img, 
            size=[new_h, new_w], 
            interpolation=InterpolationMode.BICUBIC, 
            antialias=True
        )
        img_cropped = center_crop(img_resized, output_size=[self.clip_crop_size, self.clip_crop_size])
        img_normalized = normalize(img_cropped, mean=self.clip_normalize_mean, std=self.clip_normalize_std)
        
        with torch.no_grad():
            pixel_values_on_device = img_normalized.to(self.clip_model.device)
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values_on_device)
            image_features_sequence = vision_outputs.last_hidden_state
            
        return image_features_sequence

    def forward(self, inp, current_iter=None):
        z_current = self._get_clip_features(inp)
        
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        x = self.intro(inp_padded)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)   
            encs.append(x)
            x = down(x)
        
        x = self.middle_guide_attention(x, z_current)
        x = self.middle_blks(x)
        z_current = self.middle_refine_attention(z_current, x)
        
        for decoder_blocks, up, enc_skip, guide_attn, refine_attn in zip(
            self.decoders, self.ups, encs[::-1], self.decoder_guide_attentions, self.decoder_refine_attentions
        ):
            x = up(x)
            x = x + enc_skip
            
            x = guide_attn(x, z_current)
            x = decoder_blocks(x)
            z_current = refine_attn(z_current, x)

        x = self.ending(x)
        x = x + inp_padded
        
        final_image_output = x[:, :, :H, :W]

        return final_image_output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


# Baseline
class SymUNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, 
                 enc_blk_nums=[], dec_blk_nums=[], 
                 clip_model_name="openai/clip-vit-large-patch14",
                 cross_attn_inner_dims=[48, 96, 192],
                 cross_attn_heads=[1, 2, 4],
                 middle_cross_attn_inner_dim=384,
                 middle_cross_attn_heads=8,
                 restormer_heads=[1, 2, 4],
                 restormer_middle_heads=8,
                 semantic_start_iter=0,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias', 
                 patch_sizes=[8, 8, 8], 
                 middle_patch_size=8,):
        super().__init__()
        
        self.current_iter = 0
        self.semantic_start_iter = semantic_start_iter
        self.semantic_modules_frozen = False
        
        num_stages = len(enc_blk_nums)
        assert len(cross_attn_inner_dims) == num_stages, "Length of cross_attn_inner_dims must match the number of encoder/decoder stages."
        assert len(cross_attn_heads) == num_stages, "Length of cross_attn_heads must match the number of encoder/decoder stages."
        assert len(restormer_heads) == num_stages, "Length of restormer_heads must match the number of encoder/decoder stages."
        
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoder_guide_attentions = nn.ModuleList()
        self.decoder_refine_attentions = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_heads[i], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2
            
        self.middle_blks = nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_middle_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(middle_blk_num)])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            
            stage_idx = num_stages - 1 - i
            current_restormer_heads = restormer_heads[stage_idx]
            
            self.decoders.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=current_restormer_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            
        self.padder_size = (2 ** len(self.encoders)) * 4
        
        if self.semantic_start_iter > 0:
            self.freeze_semantic_modules()

    def forward(self, inp, current_iter=None):
        if current_iter is not None:
            self._update_training_stage(current_iter)
        
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)
        x = self.intro(inp_padded)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)   
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder_blocks, up, enc_skip in zip(
            self.decoders, self.ups, encs[::-1]
        ):
            x = up(x)
            x = x + enc_skip
            
            x = decoder_blocks(x)
            
        x = self.ending(x)
        x = x + inp_padded
        
        final_image_output = x[:, :, :H, :W]

        return final_image_output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
