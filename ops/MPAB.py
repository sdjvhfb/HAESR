import  torch
from    torch import nn, einsum

from    einops import rearrange, repeat
from    einops.layers.torch import Rearrange, Reduce
import  torch.nn.functional as F
from    ops.layernorm import LayerNorm2d

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CA(nn.Module):
    def __init__(self, channels):
        super(CA, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.AdaptiveAvgPool(x))
        out = out * x
        return out


class PLKB(nn.Module):
    """Partial large-kernel block adapted from EPLKB."""

    def __init__(self, channels, large_kernel, split_group):
        super(PLKB, self).__init__()
        self.channels = channels
        self.split_group = split_group
        self.split_channels = int(channels // split_group)
        self.CA = CA(channels)
        self.DWConv_Kx1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(large_kernel, 1), stride=1,
                                    padding=(large_kernel // 2, 0), groups=self.split_channels)
        self.DWConv_1xK = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, large_kernel), stride=1,
                                    padding=(0, large_kernel // 2), groups=self.split_channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        # Shuffle channels before partial large-kernel processing.
        B, C, H, W = x.size()
        x = x.reshape(B, self.split_channels, self.split_group, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)

        x1 = self.CA(x1)

        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out


class HFB(nn.Module):
    """Hybrid Feature Block."""

    def __init__(self, channels, large_kernel, split_group):
        super(HFB, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_group)
        self.DWConv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.DWConv3(x)
        x2 = self.PLKB(x)
        out = self.act(self.conv1(torch.cat((x1, x2), dim=1)))
        return out


class ExtendedWindowUnfold(nn.Module):
    def __init__(self, window_size=8, extend_size=4, sample_pattern='star'):
        super().__init__()
        self.w = window_size
        self.ext = extend_size
        self.pattern = sample_pattern
        
        self.ext_size = window_size + 2 * extend_size
        
        self.unfold = nn.Unfold(
            kernel_size=(self.ext_size, self.ext_size),
            stride=window_size,
            padding=extend_size
        )
        
        self.register_buffer('sample_indices', self._create_sample_indices())
        self.register_buffer('center_indices', self._create_center_indices())
        
    def _create_center_indices(self):
        w = self.w
        ext = self.ext
        ew = self.ext_size
        indices = []
        for i in range(w):
            for j in range(w):
                idx = (ext + i) * ew + (ext + j)
                indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)
        
    def _create_sample_indices(self):
        """Retain the center window and add pattern-specific peripheral samples."""
        w = self.w
        ext = self.ext
        ew = self.ext_size
        
        indices = []
        
        for i in range(w):
            for j in range(w):
                idx = (ext + i) * ew + (ext + j)
                indices.append(idx)
        
        if self.pattern == 'star':
            # Star pattern along eight directions.
            col = ext + w // 2
            for step in range(ext):
                row = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext + w + step
                indices.append(row * ew + col)
            row = ext + w // 2
            for step in range(ext):
                col = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                col = ext + w + step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext - 1 - step
                col = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext - 1 - step
                col = ext + w + step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext + w + step
                col = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext + w + step
                col = ext + w + step
                indices.append(row * ew + col)
                
        elif self.pattern == 'cross':
            # Cross pattern with two central rows or columns.
            row = ext - 1
            for step in range(ext):
                r = row - step
                for col_offset in [-1, 0]:
                    c = ext + w // 2 + col_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
            
            row = ext + w
            for step in range(ext):
                r = row + step
                for col_offset in [-1, 0]:
                    c = ext + w // 2 + col_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
            
            col = ext - 1
            for step in range(ext):
                c = col - step
                for row_offset in [-1, 0]:
                    r = ext + w // 2 + row_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
            
            col = ext + w
            for step in range(ext):
                c = col + step
                for row_offset in [-1, 0]:
                    r = ext + w // 2 + row_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
                
        elif self.pattern == 'corner':
            # Sample 3x3 corner regions outside the center window.
            corner_size = 3
            
            corner_starts = [
                (0, 0),
                (0, ew - corner_size),
                (ew - corner_size, 0),
                (ew - corner_size, ew - corner_size),
            ]
            
            for start_r, start_c in corner_starts:
                for i in range(corner_size):
                    for j in range(corner_size):
                        r = start_r + i
                        c = start_c + j
                        if not (ext <= r < ext + w and ext <= c < ext + w):
                            indices.append(r * ew + c)
        
        return torch.tensor(sorted(set(indices)), dtype=torch.long)
    
    def forward(self, x):
        """Input is [B, C, H, W], with H and W divisible by window_size.

        Return center tokens [B, NW, w*w, C] for queries, extended tokens
        [B, NW, num_samples, C] for keys and values, and the window grid (nH, nW).
        """
        B, C, H, W = x.shape
        w = self.w
        ew = self.ext_size
        
        nH, nW = H // w, W // w
        NW = nH * nW
        
        # Unfolded shape [B, C*ew*ew, NW].
        unfolded = self.unfold(x)
        
        unfolded = unfolded.view(B, C, ew * ew, NW)
        unfolded = unfolded.permute(0, 3, 2, 1)
        
        center = unfolded[:, :, self.center_indices, :]
        extended = unfolded[:, :, self.sample_indices, :]
        
        return center, extended, (nH, nW)


class MMA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        extend_size=2,
        sample_pattern='star',
        with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.w = window_size
        self.with_pe = True
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        self.ext_unfold = ExtendedWindowUnfold(window_size, extend_size, sample_pattern)
        
        if self.with_pe:
            num_samples = len(self.ext_unfold.sample_indices)
            num_q = window_size * window_size
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.heads, num_q, num_samples))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
    
    def forward(self, x):
        """Return features [B, C, H, W] and attention [B*NW, heads, w*w, num_samples]."""
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        center, extended, (nH, nW) = self.ext_unfold(x)
        
        NW = nH * nW
        
        q = self.to_q(center)
        
        kv = self.to_kv(extended)
        k, v = kv.chunk(2, dim=-1)
        
        q = rearrange(q, 'b n s (h d) -> (b n) h s d', h=h)
        k = rearrange(k, 'b n s (h d) -> (b n) h s d', h=h)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        
        q = q * self.scale
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        
        if self.with_pe:
            sim = sim + self.rel_pos_bias.unsqueeze(0)
        
        attn = self.attend(sim)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=NW)
        
        out = self.to_out(out)
        
        out = rearrange(out, 'b (nh nw) (w1 w2) c -> b c (nh w1) (nw w2)', 
                       nh=nH, nw=nW, w1=w, w2=w)
        
        return out, attn


class MMAReuse(nn.Module):
    """Apply attention generated by MMA to values sampled with the same pattern."""
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        extend_size=2,
        sample_pattern='star',
    ):
        super().__init__()
        assert (dim % dim_head) == 0
        
        self.heads = dim // dim_head
        self.w = window_size
        
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        self.ext_unfold = ExtendedWindowUnfold(window_size, extend_size, sample_pattern)
    
    def forward(self, x, attn):
        """Use MMA attention with matching window size, extension and sampling pattern."""
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        _, extended, (nH, nW) = self.ext_unfold(x)
        NW = nH * nW
        
        v = self.to_v(extended)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=NW)
        
        out = self.to_out(out)
        out = rearrange(out, 'b (nh nw) (w1 w2) c -> b c (nh w1) (nw w2)',
                       nh=nH, nw=nW, w1=w, w2=w)
        
        return out


class GridCenterUnfold(nn.Module):
    """Gather grid tokens and diagonal midpoint samples."""
    def __init__(self, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_base = window_size * window_size
        self.num_extra = (window_size - 1) * (window_size - 1)
        self.num_kv = self.num_base + self.num_extra
        
    def forward(self, x):
        """Input is [B, C, H, W], with H and W divisible by window_size.

        Return queries [B*nH*nW, w*w, C], keys and values
        [B*nH*nW, w*w + (w-1)*(w-1), C], and the group grid (nH, nW).
        """
        B, C, H, W = x.shape
        w = self.w
        nH, nW = H // w, W // w
        
        x_grid = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        q_tokens = rearrange(x_grid, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        # Shift by half the grid spacing along both axes.
        offset_h, offset_w = nH // 2, nW // 2
        
        x_padded = F.pad(x, (0, offset_w, 0, offset_h), mode='reflect')
        
        x_offset = x_padded[:, :, offset_h:offset_h+H, offset_w:offset_w+W]
        
        x_center_grid = rearrange(x_offset, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        
        centers = x_center_grid[:, :, :, :w-1, :w-1, :]
        centers = rearrange(centers, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        kv_tokens = torch.cat([q_tokens, centers], dim=1)
        
        return q_tokens, kv_tokens, (nH, nW)


class GridRowMidUnfold(nn.Module):
    """Gather grid tokens and row-wise midpoint samples."""
    def __init__(self, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_base = window_size * window_size
        self.num_extra = window_size * (window_size - 1)
        self.num_kv = self.num_base + self.num_extra
        
    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        nH, nW = H // w, W // w
        
        x_grid = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        q_tokens = rearrange(x_grid, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        # Shift by half the horizontal grid spacing.
        offset_w = nW // 2
        
        x_padded = F.pad(x, (0, offset_w, 0, 0), mode='reflect')
        x_offset = x_padded[:, :, :, offset_w:offset_w+W]
        
        x_row_grid = rearrange(x_offset, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        
        row_mids = x_row_grid[:, :, :, :, :w-1, :]
        row_mids = rearrange(row_mids, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        kv_tokens = torch.cat([q_tokens, row_mids], dim=1)
        
        return q_tokens, kv_tokens, (nH, nW)


class GridColMidUnfold(nn.Module):
    """Gather grid tokens and column-wise midpoint samples."""
    def __init__(self, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_base = window_size * window_size
        self.num_extra = (window_size - 1) * window_size
        self.num_kv = self.num_base + self.num_extra
        
    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        nH, nW = H // w, W // w
        
        x_grid = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        q_tokens = rearrange(x_grid, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        # Shift by half the vertical grid spacing.
        offset_h = nH // 2
        
        x_padded = F.pad(x, (0, 0, 0, offset_h), mode='reflect')
        x_offset = x_padded[:, :, offset_h:offset_h+H, :]
        
        x_col_grid = rearrange(x_offset, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        
        col_mids = x_col_grid[:, :, :, :w-1, :, :]
        col_mids = rearrange(col_mids, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        kv_tokens = torch.cat([q_tokens, col_mids], dim=1)
        
        return q_tokens, kv_tokens, (nH, nW)


class MGA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        sample_mode='center',
        with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.w = window_size
        self.with_pe = True
        self.sample_mode = sample_mode
        
        if sample_mode == 'center':
            self.grid_unfold = GridCenterUnfold(window_size)
        elif sample_mode == 'row_mid':
            self.grid_unfold = GridRowMidUnfold(window_size)
        elif sample_mode == 'col_mid':
            self.grid_unfold = GridColMidUnfold(window_size)
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")
        
        self.num_base = self.grid_unfold.num_base
        self.num_extra = self.grid_unfold.num_extra
        self.num_kv = self.grid_unfold.num_kv

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)
            
            # Learn separate position biases for the added midpoint tokens.
            self.extra_pos_bias = nn.Parameter(torch.zeros(self.heads, self.num_base, self.num_extra))
            nn.init.trunc_normal_(self.extra_pos_bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        nH, nW = H // w, W // w
        
        q_tokens, kv_tokens, _ = self.grid_unfold(x)
        
        q = self.to_q(q_tokens)
        kv = self.to_kv(kv_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        
        if self.with_pe:
            bias_base = self.rel_pos_bias(self.rel_pos_indices)
            bias_base = rearrange(bias_base, 'i j h -> h i j')
            bias_extra = self.extra_pos_bias
            bias = torch.cat([bias_base, bias_extra], dim=-1)
            sim = sim + bias.unsqueeze(0)

        attn = self.attend(sim)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=w, w2=w)
        out = self.to_out(out)
        
        out = rearrange(out, '(b x y) w1 w2 d -> b d (w1 x) (w2 y)', x=nH, y=nW)
        return out, attn


class MGAReuse(nn.Module):
    """Apply attention generated by MGA to values from the same grid sampling pattern."""
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        sample_mode='center',
    ):
        super().__init__()
        assert (dim % dim_head) == 0

        self.heads = dim // dim_head
        self.w = window_size
        
        if sample_mode == 'center':
            self.grid_unfold = GridCenterUnfold(window_size)
        elif sample_mode == 'row_mid':
            self.grid_unfold = GridRowMidUnfold(window_size)
        elif sample_mode == 'col_mid':
            self.grid_unfold = GridColMidUnfold(window_size)
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn):
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        nH, nW = H // w, W // w
        
        _, kv_tokens, _ = self.grid_unfold(x)

        v = self.to_v(kv_tokens)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=w, w2=w)

        out = self.to_out(out)
        
        out = rearrange(out, '(b x y) w1 w2 d -> b d (w1 x) (w2 y)', x=nH, y=nW)
        return out


class Channel_Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)


        out = self.project_out(out)

        return out
        

class Channel_Attention_grid(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)


        out = self.project_out(out)

        return out


class MMABlock(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True, 
                 window_size=8, extend_size=2, sample_pattern='star',
                 with_pe=True, dropout=0.0):
        super(MMABlock, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = MMA(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, extend_size=extend_size,
            sample_pattern=sample_pattern, with_pe=with_pe
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x):
        attn_out, a = self.attn(self.cnorm(x))
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x, a


class MGABlock(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, sample_mode='center', with_pe=True, dropout=0.0):
        super(MGABlock, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = MGA(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, sample_mode=sample_mode, with_pe=with_pe
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x):
        attn_out, a = self.attn(self.cnorm(x))
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x, a


class MMAReuseBlock(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, extend_size=2, sample_pattern='star',
                 with_pe=True, dropout=0.0):
        super(MMAReuseBlock, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = MMAReuse(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, extend_size=extend_size,
            sample_pattern=sample_pattern
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x, a):
        attn_out = self.attn(self.cnorm(x), a)
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x


class MGAReuseBlock(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, sample_mode='center', with_pe=True, dropout=0.0):
        super(MGAReuseBlock, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = MGAReuse(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, sample_mode=sample_mode
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x, a):
        attn_out = self.attn(self.cnorm(x), a)
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x


class CAM(nn.Module):
    """Channel attention within windows and across the global grid."""
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(CAM, self).__init__()

        self.w= 8
        
                
        self.canorm1 = LayerNorm2d(channel_num)
        self.canorm2 = LayerNorm2d(channel_num)
        
        self.cttn1 = Channel_Attention(dim = channel_num, heads=1, dropout = dropout, window_size = window_size)
        self.cttn2 = Channel_Attention_grid(dim = channel_num, heads=1, dropout = dropout, window_size = window_size)
        
        
        self.cnorm1 = LayerNorm2d(channel_num)
        self.cnorm2 = LayerNorm2d(channel_num)
       
        self.gfn1 = Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)
        self.gfn2 = Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)

    def forward(self, x):

        x = self.cttn1(self.canorm1(x))+x
        x = self.gfn1(self.cnorm1(x))+x
        x = self.cttn2(self.canorm2(x))+x
        x = self.gfn2(self.cnorm2(x))+x

        return x


class InitialRefinementStages(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True, 
                 window_size=8, extend_size=4, sample_pattern='star',
                 mga_modes=None, with_pe=True, dropout=0.0):
        super(InitialRefinementStages, self).__init__()

        self.w = 8
        self.ch_split = channel_num // 4
        
        if mga_modes is None:
            mga_modes = ['center', 'row_mid', 'col_mid']
        
        self.HFB1 = HFB(channel_num, large_kernel=31, split_group=4)
        self.HFB2 = HFB(channel_num//4*3, large_kernel=31, split_group=4)
        self.HFB3 = HFB(channel_num//2, large_kernel=31, split_group=4)

        mma_patterns = ['star', 'cross', 'corner']
        self.MMA1 = MMABlock(channel_num//4*3, depth, bias, ffn_bias, window_size, 
                              extend_size, mma_patterns[0], with_pe, dropout)
        self.MMA2 = MMABlock(channel_num//2, depth, bias, ffn_bias, window_size,
                              extend_size, mma_patterns[1], with_pe, dropout)
        self.MMA3 = MMABlock(channel_num//4, depth, bias, ffn_bias, window_size,
                              extend_size, mma_patterns[2], with_pe, dropout)
        
        self.MGA1 = MGABlock(channel_num//4*3, depth, bias, ffn_bias, window_size,
                                  mga_modes[0], with_pe, dropout)
        self.MGA2 = MGABlock(channel_num//2, depth, bias, ffn_bias, window_size,
                                  mga_modes[1], with_pe, dropout)
        self.MGA3 = MGABlock(channel_num//4, depth, bias, ffn_bias, window_size,
                                  mga_modes[2], with_pe, dropout)
        
        self.CAM = CAM(channel_num, depth, bias, ffn_bias, window_size, with_pe, dropout)
        
        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num//4*3, channel_num//4*3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)

    def forward(self, x):
        # Stage 1 processes 4c channels, where c = channel_num // 4.
        x = self.HFB1(x)
        loc = x[:, :self.ch_split, :, :]
        x, a1 = self.MMA1(x[:, self.ch_split:, :, :])
        x, a2 = self.MGA1(x)
        
        x = self.pw1(torch.cat((x, loc), 1))
        x1 = x[:, :self.ch_split, :, :]
        
        # Stage 2 processes the remaining 3c channels.
        x = self.HFB2(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]
        x, a3 = self.MMA2(x[:, self.ch_split:, :, :])
        x, a4 = self.MGA2(x)
        
        x = self.pw2(torch.cat((x, loc), 1))
        x2 = x[:, :self.ch_split, :, :]
        
        # Stage 3 processes the remaining 2c channels.
        x = self.HFB3(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]
        x, a5 = self.MMA3(x[:, self.ch_split:, :, :])
        x, a6 = self.MGA3(x)
        
        x = self.pw3(torch.cat((x1, x2, x, loc), 1))
        x = self.CAM(x)

        return x, a1, a2, a3, a4, a5, a6


class AttentionReuseStages(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, extend_size=4, sample_pattern='star',
                 mga_modes=None, with_pe=True, dropout=0.0):
        super(AttentionReuseStages, self).__init__()

        self.w = 8
        self.ch_split = channel_num // 4
        
        if mga_modes is None:
            mga_modes = ['center', 'row_mid', 'col_mid']
        
        self.HFB1 = HFB(channel_num, large_kernel=31, split_group=4)
        self.HFB2 = HFB(channel_num//4*3, large_kernel=31, split_group=4)
        self.HFB3 = HFB(channel_num//2, large_kernel=31, split_group=4)

        mma_patterns = ['star', 'cross', 'corner']
        self.MMA1 = MMAReuseBlock(channel_num//4*3, depth, bias, ffn_bias, window_size,
                              extend_size, mma_patterns[0], with_pe, dropout)
        self.MMA2 = MMAReuseBlock(channel_num//2, depth, bias, ffn_bias, window_size,
                              extend_size, mma_patterns[1], with_pe, dropout)
        self.MMA3 = MMAReuseBlock(channel_num//4, depth, bias, ffn_bias, window_size,
                              extend_size, mma_patterns[2], with_pe, dropout)
        
        self.MGA1 = MGAReuseBlock(channel_num//4*3, depth, bias, ffn_bias, window_size,
                                  mga_modes[0], with_pe, dropout)
        self.MGA2 = MGAReuseBlock(channel_num//2, depth, bias, ffn_bias, window_size,
                                  mga_modes[1], with_pe, dropout)
        self.MGA3 = MGAReuseBlock(channel_num//4, depth, bias, ffn_bias, window_size,
                                  mga_modes[2], with_pe, dropout)
        
        self.CAM = CAM(channel_num, depth, bias, ffn_bias, window_size, with_pe, dropout)
        
        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num//4*3, channel_num//4*3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)

    def forward(self, x, a1, a2, a3, a4, a5, a6):
        # Stage 1 processes 4c channels, where c = channel_num // 4.
        x = self.HFB1(x)
        loc = x[:, :self.ch_split, :, :]
        x = self.MMA1(x[:, self.ch_split:, :, :], a1)
        x = self.MGA1(x, a2)
        
        x = self.pw1(torch.cat((x, loc), 1))
        x1 = x[:, :self.ch_split, :, :]
        
        # Stage 2 processes the remaining 3c channels.
        x = self.HFB2(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]
        x = self.MMA2(x[:, self.ch_split:, :, :], a3)
        x = self.MGA2(x, a4)
        
        x = self.pw2(torch.cat((x, loc), 1))
        x2 = x[:, :self.ch_split, :, :]
        
        # Stage 3 processes the remaining 2c channels.
        x = self.HFB3(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]
        x = self.MMA3(x[:, self.ch_split:, :, :], a5)
        x = self.MGA3(x, a6)
        
        x = self.pw3(torch.cat((x1, x2, x, loc), 1))
        x = self.CAM(x)

        return x