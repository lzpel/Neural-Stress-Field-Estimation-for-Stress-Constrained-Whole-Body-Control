ナイスキャッチです。その **AttnBlock** は “Transformer ではない” けれど、**VAE内部に入っている空間自己注意（Non-Local Block）** です。Stable Diffusion の VAE 由来の実装でも出てきます（bottleneck などの低解像度にだけ挿すやつ）。ポイントを整理します。

## 何者？Transformer との違い

* **目的**: CNN だけだと受容野が有限なので、**広域の相関（長距離依存）を 1 回で取り込む**ためのブロック。
* **違い**:

  * Multi-Head ではなく **単一ヘッド**（実装によっては複数に拡張可）。
  * FFN/残りの Transformer サブ層はない。**“非局所注意”を 1 層足しているだけ**。
  * 位置埋め込みも無し（畳み込みの座標系に乗ったまま）。

## 計算量（ここが重要）

入力特徴が (B \times C \times H \times W) のとき、

* 注意行列は **(HW \times HW)** なので時間・メモリともに **(O((HW)^2))**。
* ただし、**使うのは通常 “最も低い解像度” だけ**（例: 64→32→16→8 の **8×8 や 16×16**）なので、**総体としてはほぼ線形スケール（O(HW)）のまま**に抑えられます。

  * あなたの設計でも、`using_sa` を最下層だけに付けていて理にかなっています。

> 結論：**VAR(2024) ＝ Transformer なし（O(HW)）**。
> ただし VAE にこの **AttnBlock（非局所注意）** を “低解像度だけ” 入れるのはよくある実装で、**計算量の支配は依然ほぼ O(HW)** に留まります（最下層の (HW) が小さいため）。
> 逆に **高解像度段に入れると一気に重く（((HW)^2))** になるので避けるべき。

## コード上の注意点（小さなバグ＆改善）

1. スケーリング係数の計算

```python
self.w_ratio = int(in_channels) ** (-0.5)
```

これは **バグります**（`int ** float` は TypeError）。正しくは：

```python
import math
self.w_ratio = 1.0 / math.sqrt(in_channels)
# あるいは torch.rsqrt(torch.tensor(C, dtype=torch.float32))
```

2. メモリと数値安定性

* `w = torch.bmm(q, k)` の前に **型を fp16/bf16** にしているなら、**スケーリング→softmax** の順（今の実装でOK）を維持。
* さらにメモリ削減したいなら **ブロック分割（窓 attention）** や **畳み込みだけにする**選択肢もあり。

3. `GroupNorm` のグループ数

* `Normalize(in_channels, num_groups=32)` は **`in_channels % 32 == 0`** を満たさないと警告/性能低下。
  小さい `C` の段では `num_groups=min(32, in_channels)` に落とすと安全です。

4. アテンションの挿し方

* すでに `i_level == last` のみで `attn.append(...)` していて、forward でも `len(attn)>0` ガードを入れているので **設計はOK**。
* **中間層（`mid.attn_1`）** は解像度が一番小さいのでコスパが良い。ここだけ残すのもあり。

## もし “純VAR” に寄せたいなら

* `using_sa=False`（すべての Attn を Identity）にすると、**完全に畳み込み型**になり、**計算量は厳密に O(HW)**。
* 再構成品質が少し落ちる可能性はありますが、ボトルネック解像度が十分小さければ差は軽微なことが多いです。

## 参考の最小修正パッチ

```diff
 class AttnBlock(nn.Module):
     def __init__(self, in_channels):
         super().__init__()
         self.C = in_channels
         self.norm = Normalize(in_channels)
         self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0)
-        self.w_ratio = int(in_channels) ** (-0.5)
+        import math
+        self.w_ratio = 1.0 / math.sqrt(in_channels)
         self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
```

---

### まとめ

* あなたの疑問どおり、**AttnBlock は “Transformer 層” ではない**です。
* ただし **(HW)^2 の自己注意**をボトルネックに**一点だけ**入れて VAE のグローバル受容野を補う、というよくある実装パターン。
* **VAR(2024) 的にスケールさせたい**なら、**低解像度だけ**に留めるか **完全に外す**のが吉。
* 上の **`w_ratio` バグ修正**は必須です。
