
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def title2key(title:str)->str:
	return title[:3]
def main(out_res, out_tas):
	# --- 可視化 ---
	fig, ax = plt.subplots(figsize=(8, 4))
    # --- タスク名を収集してユニーク化 ---
	title_keys = sorted(set(title2key(t[0]) for t in out_tas))
	cmap = cm.get_cmap('tab10', len(title_keys))  # 10色パレット（必要なら Set3, tab20 もOK）
	color_map = {t: cmap(i) for i, t in enumerate(title_keys)}
	# task=[title, start_time, end_time, rank(このrankを追加する部分)] 同じrankにタスクが重複しないように貪欲に詰め合わせる
	import heapq
	active = []  # (end, rank) を end 昇順のヒープで保持
	free   = []  # 再利用可能な rank を小さい順に保持
	next_rank = 0

	for task in out_tas:
		s, e = task[1], task[2]

		# 終了済みタスクを解放（end <= s を「非重複」とみなす場合）
		while active and active[0][0] <= s:
			_, r = heapq.heappop(active)
			heapq.heappush(free, r)

		# 最小の空き rank を割り当て（なければ新規発番）
		if free:
			r = heapq.heappop(free)
		else:
			r = next_rank
			next_rank += 1

		task.append(r)
		heapq.heappush(active, (e, r))
	for i in out_tas:
		title=i[0]
		start=i[1]
		width=i[2]-start
		y=i[-1]
		print(y)
		ax.barh(
			y=y, 
			width=width, 
			left=start, 
			height=0.4, 
			align='center', 
			color=color_map[title2key(title)], 
			#edgecolor='black'
		)
		# ax.text(start, y, title, va='center')

	# --- 軸設定 ---
	ax.set_yticks([])
	ax.set_xlabel("Time [s]")
	ax.set_title("Server Task Timeline (Overlap Visualization)")
	# ---凡例---
	task_handles = [
		mpatches.Patch(color=color_map[title2key(k)], label=k)
		for k in ["animation", "render", "archive", "transport"]#title_keys
	]
	# ax.legend(handles=task_handles, title="Task")
	# ===== ここから VRAM ラインを重ねる =====
	# out_res: [[time_sec, vram_gb], ...] を想定
	if out_res:
		resouce = np.array([[i, *row] for i, row in enumerate(out_res)])

		# 右 y 軸を作成（x は共有）
		ax2 = ax.twinx()
		ax2.plot(
			resouce[:, 0], resouce[:, 1], label="VRAM [GB]"
		)
		ax2.plot(
			resouce[:, 0], resouce[:, 2], label="Ideal transfer [Gbps]"
		)
	# 折れ線のハンドルは ax2 から取得
	line_handles, line_labels = ax2.get_legend_handles_labels()
	ax.legend(
		handles=task_handles + line_handles,
		labels=title_keys + line_labels,
		title="Task / Resource",
		loc="upper right"
	)

	plt.tight_layout()
	plt.show()
def read(path: str, index_str:int):
	data = []
	with open(path) as f:
		for line in f:
			parts = [v if i<index_str else int(v)  for i,v in enumerate(line.strip().split())]
			data.append(parts)
	return data
if __name__=="__main__":
	out_res=read("out.res.csv", 0)
	out_tas=read("out.tas.csv", 1)
	main(out_res, out_tas)