import pandas as pd

# 读取文件
df = pd.read_csv("datasets/2019-20.csv")

# 转换日期格式（如果是字符串格式）
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.strftime("%d/%m/%y")

# 保存回原文件或新文件
df.to_csv("datasets/2019-20.csv", index=False)
print("✅ 日期格式已修改为两位年份！")