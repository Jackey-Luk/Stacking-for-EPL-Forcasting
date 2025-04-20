import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import os
import uuid
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 加载模型
lstm_model = load_model('models/lstm_model.keras')
xgboost_model = joblib.load('models/xgboost_model.pkl')
lightgbm_model = joblib.load('models/lightgbm_model.pkl')
catboost_model = joblib.load('models/catboost_model.pkl')

# 读取测试数据
dataTestsouce = pd.read_csv("datasets/allAtt_onehot_large_test_new8.csv")

# Flask 实例
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


# 结果解释
def interpret_result(val):
    return '主队胜' if val == 0 else '主队非胜'


# 主客队历史交锋图表
def create_match_bar_chart(df, home_team, away_team):
    plt.figure(figsize=(10, 6))
    dates = df['Date'].dt.strftime('%Y-%m-%d')
    x = np.arange(len(dates))
    home_goals, away_goals = [], []

    for _, row in df.iterrows():
        if row['HomeTeam'] == home_team:
            home_goals.append(row['FTHG'])
            away_goals.append(row['FTAG'])
        else:
            home_goals.append(row['FTAG'])  # 主客反转
            away_goals.append(row['FTHG'])

    width = 0.35
    plt.bar(x - width / 2, home_goals, width=width, label=home_team)
    plt.bar(x + width / 2, away_goals, width=width, label=away_team)

    plt.xticks(x, dates, rotation=45)
    plt.xlabel('比赛日期')
    plt.ylabel('进球数')
    plt.title(f'{home_team} vs {away_team} 历史交战进球数')
    plt.legend()
    plt.tight_layout()

    filename = f'static/history_chart_{uuid.uuid4().hex}.png'
    plt.savefig(filename)
    plt.close()
    return filename


# 主队或客队历史进球图
def create_team_history_list(team_matches, team):
    team_matches = team_matches.sort_values(by='Date', ascending=False).head(5)  # 最近5场
    team_matches = team_matches.sort_values(by='Date')  # 为了按时间顺序显示
    history_list = []

    for _, row in team_matches.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        if row['HomeTeam'] == team:
            line = f"{date}_{team}_{row['FTHG']} ：{row['FTAG']}_{row['AwayTeam']}"
        else:
            line = f"{date}_{row['HomeTeam']}_{row['FTHG']} ：{row['FTAG']}_{team}"
        history_list.append(line)

    return history_list

@app.route('/predict', methods=['POST'])
def predict():
    global home_team_history_list, away_team_history_list
    if request.method == 'POST':
        date = request.form['date']
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        model_choice = request.form['model_choice']

        match_data = dataTestsouce[
            (dataTestsouce['Date'] == date) &
            (dataTestsouce['HomeTeam'] == home_team) &
            (dataTestsouce['AwayTeam'] == away_team)
        ]

        if match_data.empty:
            return render_template('index.html', error=f"没有这场比赛：{home_team} vs {away_team} 在 {date}",
                                   date=date, home_team=home_team, away_team=away_team, model_choice=model_choice)

        x_test = match_data.iloc[:, 4:38].values
        y_test = match_data.iloc[:, 38:].values

        if model_choice == 'lstm':
            x_test = np.reshape(x_test, (x_test.shape[0], 34, 1))

        if model_choice == 'lstm':
            prediction = lstm_model.predict(x_test)
            prediction = np.argmax(prediction, axis=1)
        elif model_choice == 'xgboost':
            prediction = xgboost_model.predict(x_test)
        elif model_choice == 'lightgbm':
            prediction = lightgbm_model.predict(x_test)
        elif model_choice == 'catboost':
            prediction = catboost_model.predict(x_test)
        else:
            return render_template('index.html', error="无效的模型选择",
                                   date=date, home_team=home_team, away_team=away_team, model_choice=model_choice)

        actual_result = np.argmax(y_test, axis=1)
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'predicted': interpret_result(prediction[0]),
            'actual': interpret_result(actual_result[0])
        }

        # 图表相关
        history_chart_path = None
        home_team_chart = None
        away_team_chart = None
        home_total_goals = 0
        away_total_goals = 0

        home_history_goals = 0
        home_history_loss = 0

        away_history_goals = 0
        away_history_loss = 0
        try:
            history_data = pd.read_csv("datasets/all_seasons.csv")
            history_data['Date'] = pd.to_datetime(history_data['Date'])
            match_date = pd.to_datetime(date)

            history_matches = history_data[
                (history_data['Date'] <= match_date) & (
                    ((history_data['HomeTeam'] == home_team) & (history_data['AwayTeam'] == away_team)) |
                    ((history_data['HomeTeam'] == away_team) & (history_data['AwayTeam'] == home_team))
                )
            ]

            history_matches = history_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]

            if not history_matches.empty:
                history_chart_path = create_match_bar_chart(history_matches, home_team, away_team)

            # 统计总进球
            for _, row in history_matches.iterrows():
                if row['HomeTeam'] == home_team:
                    home_total_goals += row['FTHG']
                    away_total_goals += row['FTAG']
                else:
                    home_total_goals += row['FTAG']
                    away_total_goals += row['FTHG']

            # 主队历史记录
            history_home_data = history_data[
                (history_data['Date'] < match_date) & (
                    (history_data['HomeTeam'] == home_team) |
                    (history_data['AwayTeam'] == home_team)
                )
            ]
            home_5 = history_home_data.sort_values(by='Date', ascending=False).head(5)  # 最近5场
            # 统计主队进球数和失球数
            for _, row in home_5.iterrows():
                if row['HomeTeam'] == home_team:
                    home_history_goals += row['FTHG']
                    home_history_loss += row['FTAG']
                else:
                    home_history_goals += row['FTAG']
                    home_history_loss += row['FTHG']
            # 客队历史记录
            history_away_data = history_data[
                (history_data['Date'] < match_date) & (
                    (history_data['HomeTeam'] == away_team) |
                    (history_data['AwayTeam'] == away_team)
                )
            ]
            away_5 = history_away_data.sort_values(by='Date', ascending=False).head(5)  # 最近5场
            # 统计主队进球数和失球数
            for _, row in away_5.iterrows():
                if row['HomeTeam'] == away_team:
                    away_history_goals += row['FTHG']
                    away_history_loss += row['FTAG']
                else:
                    away_history_goals += row['FTAG']
                    away_history_loss += row['FTHG']
            home_team_history_list = create_team_history_list(history_home_data, home_team)
            away_team_history_list = create_team_history_list(history_away_data, away_team)
            print(home_team_history_list)
            print(away_team_history_list)

        except Exception as e:
            print("生成图表出错：", e)

        return render_template('index.html',
                               result=result,
                               date=date,
                               home_team=home_team,
                               away_team=away_team,
                               model_choice=model_choice,
                               history_chart=history_chart_path,
                               home_total_goals=home_total_goals,
                               away_total_goals=away_total_goals,
                               home_team_history_list=home_team_history_list,
                               away_team_history_list=away_team_history_list,
                               home_history_goals=home_history_goals,
                               home_history_loss=home_history_loss,
                               away_history_goals=away_history_goals,
                               away_history_loss=away_history_loss
                               )


if __name__ == '__main__':
    app.run(debug=True)
