
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def main(tasks):
	# --- 可視化 ---
	fig, ax = plt.subplots(figsize=(8, 4))
    # --- タスク名を収集してユニーク化 ---
	titles = sorted(set(t[-1] for t in tasks))
	cmap = cm.get_cmap('tab10', len(titles))  # 10色パレット（必要なら Set3, tab20 もOK）
	color_map = {t: cmap(i) for i, t in enumerate(titles)}

	for i in tasks:
		y=int(i[-2])
		start=int(i[0])
		width=int(i[1])
		title=i[-1]
		ax.barh(
			y=y, 
			width=width, 
			left=start, 
			height=0.4, 
			align='center', 
			color=color_map[title], 
			#edgecolor='black'
		)
		# ax.text(start, y, title, va='center')

	# --- 軸設定 ---
	ax.set_yticks([])
	ax.set_xlabel("Time [s]")
	ax.set_title("Server Task Timeline (Overlap Visualization)")

	plt.tight_layout()
	plt.show()
def read(path: str):
	data = []
	with open(path) as f:
		for line in f:
			parts = line.strip().split()
			data.append(parts)
	return data
if __name__=="__main__":
	data=read("out.csv")
	main(data)