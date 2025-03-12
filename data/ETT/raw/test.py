import matplotlib.pyplot as plt
import pandas as pd

data = {
    "模型": ["LSTM", "Autoforer", "ARMA-TSF"],
    "MAE(24h)": [0.3742, 0.2815, 0.2508],
    "RMSE(24h)": [0.5893, 0.4972, 0.4661],
    "参数量(M)": [1.25, 2.15, 0.89],
    "推理时延(ms)": [12.5, 18.1, 10.2]
}
df = pd.DataFrame(data)

# 生成表格图片
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="left",
    loc="center",
    colColours=["#f5f5f5"]*len(df.columns)  # 标题行背景色
)
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.savefig("table.png", dpi=300, bbox_inches="tight")