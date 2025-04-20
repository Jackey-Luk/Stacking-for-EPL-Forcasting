import pandas as pd
from datetime import datetime

all_data = []

# 循环读取文件
for year in range(2010, 2024):  # 到2022-23赛季
    next_year = str(year + 1)[-2:]  # 获取后一年的后两位
    filename = f"{year}-{next_year}.csv"

    try:
        df = pd.read_csv(filename)

        # 处理日期：转换为标准 yyyy-mm-dd 格式
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')  # 自动识别年份
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # 转换格式为字符串

        all_data.append(df)
    except Exception as e:
        print(f"读取 {filename} 时出错: {e}")

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 保存新文件
combined_df.to_csv('all_seasons.csv', index=False)

print("✅ 合并完成并已保存为 all_seasons.csv")