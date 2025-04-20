import pandas as pd
import glob
import os

# 积分规则
def compute_standings(df):
    teams = {}
    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        ftr = row['FTR']
        fthg = row['FTHG']
        ftag = row['FTAG']

        for team in [home, away]:
            if team not in teams:
                teams[team] = {'Points': 0, 'GD': 0}  # GD: goal difference

        if ftr == 'H':
            teams[home]['Points'] += 3
        elif ftr == 'D':
            teams[home]['Points'] += 1
            teams[away]['Points'] += 1
        elif ftr == 'A':
            teams[away]['Points'] += 3

        teams[home]['GD'] += fthg - ftag
        teams[away]['GD'] += ftag - fthg

    # 排序
    sorted_teams = sorted(teams.items(), key=lambda x: (-x[1]['Points'], -x[1]['GD']))
    return {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

# 初始化结果字典
ranking_df = pd.DataFrame()

# 遍历所有 CSV 文件
path = "dataset"  # 放CSV的文件夹路径
for file in sorted(glob.glob(os.path.join(path, "*.csv")), reverse=True):
    season = file[-11:-4]  # 文件名中提取赛季，如 "2023-24"
    df = pd.read_csv(file)
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
    standings = compute_standings(df)

    season_year = season[:4][-2:]  # 取“23”表示2023赛季
    season_series = pd.Series(standings, name=season_year)
    ranking_df = pd.concat([ranking_df, season_series], axis=1)

# 填充缺失值为NaN（某些赛季不在英超）
ranking_df = ranking_df.sort_index()
ranking_df.index.name = "Team"
ranking_df = ranking_df.sort_index(axis=1, ascending=False)  # 从最近赛季往前排

ranking_df.to_csv("team_season_rankings.csv")
print("✅ 每个赛季球队排名表已生成：team_season_rankings.csv")
