"""LadaGAN model for PyTorch.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_upsample(x, H, W):
    B, N, C = x.shape
    assert N == H*W
    x = x.view(B, H, W, C)
    x = x.view(B, H, W, C//4, 2, 2).permute(0,3,1,4,2,5).contiguous()
    x = x.view(B, C//4, H*2, W*2)
    B, C, H, W = x.shape
    return x, H, W, C


class SMLayerNormalization(nn.Module):
    def __init__(self, epsilon=1e-6, initializer='orthogonal'):
        super(SMLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.initializer = initializer
        
    def build(self, inputs):
        input_shape, _ = inputs
        self.h = nn.Linear(input_shape[2], input_shape[2])
        self.gamma = nn.Linear(input_shape[2], input_shape[2])
        self.beta = nn.Linear(input_shape[2], input_shape[2])
        self.ln = nn.LayerNorm(input_shape[2], elementwise_affine=False)

        # Initialize weights if using orthogonal
        if self.initializer == 'orthogonal':
            nn.init.orthogonal_(self.h.weight)
            nn.init.orthogonal_(self.gamma.weight)
            nn.init.orthogonal_(self.beta.weight)

    def forward(self, inputs):
        x, z = inputs
        x = self.ln(x)
        h = self.h(z)
        h = F.relu(h)
        
        scale = self.gamma(h)
        shift = self.beta(h)
        x = x * scale.unsqueeze(1)
        x = x + shift.unsqueeze(1)
        return x


class AdditiveAttention(nn.Module):
    def __init__(self, model_dim, n_heads, initializer='orthogonal'):
        super(AdditiveAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        
        self.q_attn = nn.Linear(model_dim, n_heads)
        dim_head = model_dim // n_heads

        self.to_out = nn.Linear(model_dim, model_dim)

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            nn.init.orthogonal_(self.wq.weight)
            nn.init.orthogonal_(self.wk.weight)
            nn.init.orthogonal_(self.wv.weight)
            nn.init.orthogonal_(self.q_attn.weight)
            nn.init.orthogonal_(self.to_out.weight)

    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        B = q.size(0)
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  
        attn = self.q_attn(q).transpose(1,2) / self.depth ** 0.5
        attn = F.softmax(attn, dim=-1)  
   
        q = self.split_into_heads(q, B)  
        k = self.split_into_heads(k, B)  
        v = self.split_into_heads(v, B)

        # calculate global vector
        global_q = torch.einsum('b h n, b h n d -> b h d', attn, q) 
        global_q = global_q.unsqueeze(2)
       
        p = global_q * k 
        r = p * v

        r = r.permute(0, 2, 1, 3)
        original_size_attention = r.reshape(B, -1, self.model_dim)

        output = self.to_out(original_size_attention)
        return output, attn


class SMLadaformer(nn.Module):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='orthogonal'):
        super(SMLadaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = SMLayerNormalization(epsilon=eps, initializer=initializer)
        self.norm2 = SMLayerNormalization(epsilon=eps, initializer=initializer)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)

    def forward(self, x, training=True):
        inputs, z = x
        x_norm1 = self.norm1([inputs, z])
        
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output) 
        
        x_norm2 = self.norm2([attn_output, z])
        mlp_output = self.mlp(x_norm2)
        return self.drop2(mlp_output), attn_maps 
    
    
class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches, model_dim, initializer='orthogonal'):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = nn.Embedding(n_patches, model_dim)

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            nn.init.orthogonal_(self.position_embedding.weight)

    def forward(self, patches):
        positions = torch.arange(self.n_patches, device=patches.device)
        return patches + self.position_embedding(positions)
    

class Generator(nn.Module):
    def __init__(self, img_size=32, model_dim=[1024, 256, 64], heads=[2, 2, 2], 
                 mlp_dim=[2048, 1024, 512], initializer='orthogonal', dec_dim=False):
        super(Generator, self).__init__()
        self.init = nn.Sequential(
            nn.Linear(model_dim[0], 8 * 8 * model_dim[0], bias=False),
            nn.Unflatten(1, (8 * 8, model_dim[0]))
        )     
    
        self.pos_emb_8 = PositionalEmbedding(64, model_dim[0], initializer=initializer)
        self.block_8 = SMLadaformer(model_dim[0], heads[0], mlp_dim[0], initializer=initializer)
        self.conv_8 = nn.Conv2d(model_dim[0], model_dim[1], 3, padding=1)

        self.pos_emb_16 = PositionalEmbedding(256, model_dim[1], initializer=initializer)
        self.block_16 = SMLadaformer(model_dim[1], heads[1], mlp_dim[1], initializer=initializer)
        self.conv_16 = nn.Conv2d(model_dim[1], model_dim[2], 3, padding=1)

        self.pos_emb_32 = PositionalEmbedding(1024, model_dim[2], initializer=initializer)
        self.block_32 = SMLadaformer(model_dim[2], heads[2], mlp_dim[2], initializer=initializer)

        self.dec_dim = dec_dim
        if self.dec_dim:
            layers = []
            for dim in self.dec_dim:
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(model_dim[2], dim, 3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(0.2)
                ])
            self.dec = nn.Sequential(*layers)
        else:
            self.patch_size = img_size // 32
            
        self.ch_conv = nn.Conv2d(model_dim[2] if not dec_dim else dec_dim[-1], 3, 3, padding=1)

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.orthogonal_(m.weight)

    def forward(self, z):
        B = z.size(0)
   
        x = self.init(z)
        x = self.pos_emb_8(x)
        x, attn_8 = self.block_8([x, z])

        x, H, W, C = pixel_upsample(x, 8, 8)
        x = self.conv_8(x)
        x = x.permute(0,2,3,1).reshape(B, H * W, -1)

        x = self.pos_emb_16(x)
        x, attn_16 = self.block_16([x, z])

        x, H, W, C = pixel_upsample(x, H, W)
        x = self.conv_16(x)
        x = x.permute(0,2,3,1).reshape(B, H * W, -1)
        x = self.pos_emb_32(x)
        x, attn_32 = self.block_32([x, z])

        x = x.view(B, 32, 32, -1).permute(0,3,1,2)
        if self.dec_dim:
            x = self.dec(x)
        elif self.patch_size != 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return [self.ch_conv(x), [attn_8, attn_16, attn_32]]

    
class DownBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, strides=2, initializer='orthogonal'):
        super(DownBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(filters//2, filters, kernel_size=kernel_size, 
                padding=kernel_size//2, stride=strides, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters, kernel_size=3, 
                padding=1, stride=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(kernel_size=strides),
            nn.Conv2d(filters//2, filters, kernel_size=1, 
                padding=0, stride=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2)
        )

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight)

    def forward(self, x):
        return (self.main(x) + self.direct(x)) / 2


class Ladaformer(nn.Module):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='orthogonal'):
        super(Ladaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim, eps=eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=eps)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)

    def forward(self, inputs, training=True):
        x_norm1 = self.norm1(inputs)
        
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output)
        
        x_norm2 = self.norm2(attn_output)
        mlp_output = self.mlp(x_norm2)
        return self.drop2(mlp_output) + attn_output, attn_maps

    
class Discriminator(nn.Module):
    def __init__(self, img_size=32, enc_dim=[64, 128, 256], out_dim=[512, 1024], mlp_dim=512, 
                 heads=2, initializer='orthogonal'):
        super(Discriminator, self).__init__()
        if img_size == 32:
            assert len(enc_dim) == 2, "Incorrect length of enc_dim for img_size 32"
        elif img_size == 64:
            assert len(enc_dim) == 3, "Incorrect length of enc_dim for img_size 64"
        elif img_size == 128:
            assert len(enc_dim) == 4, "Incorrect length of enc_dim for img_size 128"
        elif img_size == 256:
            assert len(enc_dim) == 5, "Incorrect length of enc_dim for img_size 256"
        else:
            raise ValueError(f"img_size = {img_size} not supported")
            
        self.enc_dim = enc_dim
        self.inp_conv = nn.Sequential(
            nn.Conv2d(3, enc_dim[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )    
        self.encoder = nn.ModuleList([DownBlock(
            i, kernel_size=3, strides=2, initializer=initializer
        ) for i in enc_dim[1:]])

        self.pos_emb_8 = PositionalEmbedding(256, enc_dim[-1], initializer=initializer)
        self.block_8 = Ladaformer(enc_dim[-1], heads, mlp_dim, initializer=initializer)
        
        self.conv_4 = nn.Conv2d(enc_dim[-1], out_dim[0], 3, padding=1)
        self.down_4 = nn.Sequential(
            nn.Conv2d(out_dim[0], out_dim[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim[1], 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

        # Initialize weights if using orthogonal
        if initializer == 'orthogonal':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight)
    
    def forward(self, img):
        x = self.inp_conv(img)
        for encoder in self.encoder:
            x = encoder(x)

        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(B, H * W, C)
        x = self.pos_emb_8(x)
        x, maps_16 = self.block_8(x)

        x = x.view(B, H, W, C).permute(0,3,1,2)
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_4(x)

        x = self.down_4(x)
        return [x.flatten(1)]