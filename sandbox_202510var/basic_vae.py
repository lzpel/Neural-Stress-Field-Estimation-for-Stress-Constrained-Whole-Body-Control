import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import torchinfo

# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder',]


"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""
# swish
def nonlinearity(x):
	return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
	return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# このモジュールに入力された特徴マップのサイズがCxHxW だとすると、出力のサイズは以下のようになります。入力サイズ: CxHxW補間後: Cx(2H)x(2W)畳み込み後: Cx(2H)x(2W)
class Upsample2x(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
	def forward(self, x):
		return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
	def forward(self, x):
		return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
	def __init__(self, *, in_channels, out_channels=None, dropout): # conv_shortcut=False,  # conv_shortcut: always False in VAE
		super().__init__()
		self.in_channels = in_channels
		out_channels = in_channels if out_channels is None else out_channels
		self.out_channels = out_channels
		
		self.norm1 = Normalize(in_channels)
		self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.norm2 = Normalize(out_channels)
		self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
		self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
		if self.in_channels != self.out_channels:
			self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		else:
			self.nin_shortcut = nn.Identity()
	
	def forward(self, x):
		h = self.conv1(F.silu(self.norm1(x), inplace=True))
		h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
		return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.C = in_channels
		
		self.norm = Normalize(in_channels)
		self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0)
		self.w_ratio = int(in_channels) ** (-0.5)
		self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
	
	def forward(self, x):
		qkv = self.qkv(self.norm(x))
		B, _, H, W = qkv.shape  # should be B,3C,H,W
		C = self.C
		q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
		
		# compute attention
		q = q.view(B, C, H * W).contiguous()
		q = q.permute(0, 2, 1).contiguous()	 # B,HW,C
		k = k.view(B, C, H * W).contiguous()	# B,C,HW
		w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW	w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
		w = F.softmax(w, dim=2)
		
		# attend to values
		v = v.view(B, C, H * W).contiguous()
		w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
		h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
		h = h.view(B, C, H, W).contiguous()
		
		return x + self.proj_out(h)


def make_attn(in_channels, using_sa=True):
	return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
	def __init__(
		self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
		dropout=0.0, in_channels=3,
		z_channels, double_z=False, using_sa=True, using_mid_sa=True,
	):
		super().__init__()
		self.ch = ch
		self.num_resolutions = len(ch_mult)
		self.downsample_ratio = 2 ** (self.num_resolutions - 1)
		self.num_res_blocks = num_res_blocks
		self.in_channels = in_channels
		
		# downsampling
		self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
		
		in_ch_mult = (1,) + tuple(ch_mult)
		self.down = nn.ModuleList()
		for i_level in range(self.num_resolutions):
			block = nn.ModuleList()
			attn = nn.ModuleList()
			block_in = ch * in_ch_mult[i_level]
			block_out = ch * ch_mult[i_level]
			for i_block in range(self.num_res_blocks):
				block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
				block_in = block_out
				if i_level == self.num_resolutions - 1 and using_sa:
					attn.append(make_attn(block_in, using_sa=True))
			down = nn.Module()
			down.block = block
			down.attn = attn
			if i_level != self.num_resolutions - 1:
				down.downsample = Downsample2x(block_in)
			self.down.append(down)
		
		# middle
		self.mid = nn.Module()
		self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
		self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
		self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
		
		# end
		self.norm_out = Normalize(block_in)
		self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
	
	def forward(self, x):
		# downsampling
		h = self.conv_in(x)
		for i_level in range(self.num_resolutions):
			for i_block in range(self.num_res_blocks):
				h = self.down[i_level].block[i_block](h)
				if len(self.down[i_level].attn) > 0:
					h = self.down[i_level].attn[i_block](h)
			if i_level != self.num_resolutions - 1:
				h = self.down[i_level].downsample(h)
		
		# middle
		h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
		
		# end
		h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
		return h


class Decoder(nn.Module):
	def __init__(
		self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
		dropout=0.0, in_channels=3,  # in_channels: raw img channels
		z_channels, using_sa=True, using_mid_sa=True,
	):
		super().__init__()
		self.ch = ch
		self.num_resolutions = len(ch_mult)
		self.num_res_blocks = num_res_blocks
		self.in_channels = in_channels
		
		# compute in_ch_mult, block_in and curr_res at lowest res
		in_ch_mult = (1,) + tuple(ch_mult)
		block_in = ch * ch_mult[self.num_resolutions - 1]
		
		# z to block_in
		self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
		
		# middle
		self.mid = nn.Module()
		self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
		self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
		self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
		
		# upsampling
		self.up = nn.ModuleList()
		for i_level in reversed(range(self.num_resolutions)):
			block = nn.ModuleList()
			attn = nn.ModuleList()
			block_out = ch * ch_mult[i_level]
			for i_block in range(self.num_res_blocks + 1):
				block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
				block_in = block_out
				if i_level == self.num_resolutions-1 and using_sa:
					attn.append(make_attn(block_in, using_sa=True))
			up = nn.Module()
			up.block = block
			up.attn = attn
			if i_level != 0:
				up.upsample = Upsample2x(block_in)
			self.up.insert(0, up)  # prepend to get consistent order
		
		# end
		self.norm_out = Normalize(block_in)
		self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)
	
	def forward(self, z):
		# z to block_in
		# middle
		h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
		
		# upsampling
		for i_level in reversed(range(self.num_resolutions)):
			for i_block in range(self.num_res_blocks + 1):
				h = self.up[i_level].block[i_block](h)
				if len(self.up[i_level].attn) > 0:
					h = self.up[i_level].attn[i_block](h)
			if i_level != 0:
				h = self.up[i_level].upsample(h)
		
		# end
		h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
		return h
def test_encoder_decoder_roundtrip_shapes():
	# 64→32（Encoder）→64（Decoder）
	B, H0, W0 = 1, 64, 64
	ch_mult = (1,2)
	z_channels = 48

	enc = Encoder(ch=64, ch_mult=ch_mult, num_res_blocks=1, dropout=0.0,
				  in_channels=3, z_channels=z_channels, double_z=False,
				  using_sa=True, using_mid_sa=True)
	dec = Decoder(ch=64, ch_mult=ch_mult, num_res_blocks=1, dropout=0.0,
				  in_channels=3, z_channels=z_channels,
				  using_sa=True, using_mid_sa=True)

	x = torch.randn(B, 3, H0, W0)
	print("\n--- モデル形状のテスト結果 ---")
	print(f"入力画像 (x) の形状: {x.shape}")
	z = enc(x)
	print(f"エンコーダ出力 (z) の形状 (潜在空間): {z.shape}")
	# VQ を挟まない “連続版” の往復（本来はここで量子化する）
	y = dec(z)
	for scale in range(1,4):
		torchinfo.summary(enc, input_size=(1, 3, 64*scale, 64*scale))
		torchinfo.summary(dec, input_size=(1, z_channels, 64*scale, 64*scale))
	print(f"デコーダ出力 (y) の形状 (再構築画像): {y.shape}")
	assert y.shape == x.shape
	"""
	--- モデル形状のテスト結果 ---
	入力画像 (x) の形状: torch.Size([1, 3, 64, 64])
	エンコーダ出力 (z) の形状 (潜在空間): torch.Size([1, 48, 32, 32])
	デコーダ出力 (y) の形状 (再構築画像): torch.Size([1, 3, 64, 64])
	"""
@pytest.mark.parametrize("train_tar_gz, test_tar_gz", [("./out/n02107312.tar.gz", "./out/n02704792.tar.gz")])
def test_learning(train_tar_gz, test_tar_gz):
	# train_tar_gz jpg画像がたくさん入っているtarへのパス
	# test_tar_gz jpg画像がたくさん入っているtarへのパス
	# エンコーダーデコーダーで元の画像に戻すというタスクを学習してください
	import io, tarfile, random
	from PIL import Image
	from torch.utils.data import Dataset, DataLoader

	torch.manual_seed(0)
	random.seed(0)
	IMG_SIZE = 64
	MAX_TRAIN_IMAGES = 256
	MAX_TEST_IMAGES = 64
	BATCH_SIZE = 16
	EPOCHS = 1
	MAX_STEPS = 200		  # 学習ステップ上限（早めに終わる）
	LR = 1e-3
	Z_CH = 48
	CH_MULT = (1, 2)		 # 64 -> 32（downsample x2）
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	print({"DEVICE": DEVICE})

	def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
		img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
		x = torch.from_numpy(
			(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
			 .view(IMG_SIZE, IMG_SIZE, 3)
			 .numpy())
		).float() / 255.0
		# (H,W,3) -> (3,H,W)
		return x.permute(2, 0, 1).contiguous()

	class TarImageDataset(Dataset):
		def __init__(self, tar_path, max_images):
			self.samples = []
			with tarfile.open(tar_path, "r:*") as tf:
				members = [m for m in tf.getmembers()
						   if m.isfile() and any(m.name.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png"))]
				# 安定性のため固定順＆最大数を制限
				members = sorted(members, key=lambda m: m.name)[:max_images]
				for m in members:
					with tf.extractfile(m) as f:
						if f is None:
							continue
						try:
							img = Image.open(io.BytesIO(f.read()))
							self.samples.append(_pil_to_tensor(img))
						except Exception:
							# 壊れた画像はスキップ
							continue
			if len(self.samples) == 0:
				raise RuntimeError(f"No images found in {tar_path}")

		def __len__(self):
			return len(self.samples)

		def __getitem__(self, idx):
			x = self.samples[idx]
			return x

	train_ds = TarImageDataset(train_tar_gz, MAX_TRAIN_IMAGES)
	test_ds  = TarImageDataset(test_tar_gz,  MAX_TEST_IMAGES)
	train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
	test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

	# モデル作成（Encoder/Decoderは上で定義済みのクラスを使用）
	enc = Encoder(ch=64, ch_mult=CH_MULT, num_res_blocks=1, dropout=0.0,
				  in_channels=3, z_channels=Z_CH, double_z=False,
				  using_sa=True, using_mid_sa=True).to(DEVICE)
	dec = Decoder(ch=64, ch_mult=CH_MULT, num_res_blocks=1, dropout=0.0,
				  in_channels=3, z_channels=Z_CH,
				  using_sa=True, using_mid_sa=True).to(DEVICE)

	opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

	def eval_loss(dataloader) -> float:
		enc.eval(); dec.eval()
		total, count = 0.0, 0
		with torch.no_grad():
			for x in dataloader:
				x = x.to(DEVICE)
				z = enc(x)
				y = dec(z)
				loss = F.l1_loss(y, x, reduction="mean")
				bs = x.size(0)
				total += loss.item() * bs
				count += bs
		return total / max(1, count)

	# 学習前のテスト損失
	test_before = eval_loss(test_dl)

	# 学習ループ（早めに切り上げる）
	enc.train(); dec.train()
	steps = 0
	for _ in range(EPOCHS):
		for x in train_dl:
			x = x.to(DEVICE)
			z = enc(x)
			y = dec(z)
			loss = F.l1_loss(y, x, reduction="mean")
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()

			steps += 1
			if steps >= MAX_STEPS:
				break
		if steps >= MAX_STEPS:
			break

	# 学習後のテスト損失
	test_after = eval_loss(test_dl)

	print(f"\n[VAE training] test L1 before: {test_before:.4f}  after: {test_after:.4f}")

	# 改善の検証（緩めの基準と絶対値の保険）
	assert (test_after < test_before) or (test_after < 0.05), f"reconstruction did not improve enough: before={test_before:.4f}, after={test_after:.4f}"