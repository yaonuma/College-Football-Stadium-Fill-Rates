import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import re


class CFFillRate:


    def __init__(self, attendance_data_path_raw_github, rankings_data_path_github, rankings_path_raw_github, team='National'):
        self.team_key = {'Arkansas State': 'Arkansas St', 'Ball State': 'Ball St', 'Boise State': 'Boise St',
                         'FIU': 'Florida Intl',
                         'Florida State': 'Florida St', 'Georgia State': 'Georgia St', 'Iowa State': 'Iowa St',
                         'Kansas State': 'Kansas St',
                         'Kent State': 'Kent', 'Miami (OH)': 'Miami OH', 'Michigan State': 'Michigan St',
                         'Middle Tennessee': 'MTSU',
                         'Northern Illinois': 'N Illinois', 'Ole Miss': 'Mississippi', 'Oregon State': 'Oregon St',
                         'Penn State': 'Penn St',
                         'San Diego State': 'San Diego St', 'UMass': 'Massachusetts', 'Western Kentucky': 'WKU'}
        self.team_key_inv = {v: k for k, v in self.team_key.items()}
        self.team = team
        self.att_path = attendance_data_path_raw_github
        self.rankings_path = rankings_path_raw_github
        self.rankings_path_gh = rankings_data_path_github
        self.dfs = self._build_dfs(self.att_path, self.rankings_path, self.rankings_path_gh, team)
        self.confidence = 0.95

    def _name_filter(self, raw):
        try:
            clean = self.team_key[raw.replace('*', '').strip()]
        except KeyError:
            clean = raw.replace('*', '').strip()
        return clean

    def _get_rank(self, row):
        try:
            rank = self.df_rank.loc[row[1], row[19]]['Ordinal Ranking']
        except KeyError:
            rank = np.nan
        return rank

    def _get_opponent_rank(self, row):
        try:
            rank = self.df_rank.loc[row[3], row[19]]['Ordinal Ranking']
        except KeyError:
            rank = np.nan
        return rank

    def _get_rank_delta(self, row):
        return np.abs(row[19] - row[20])

    def _get_rank_average(self, row):
        return np.mean([row[19], row[20]])

    def _convert_time(self, x):
        hour = x.split(' ')[0].split(':')
        time = float(hour[0]) + 12 + float(hour[1]) / 60
        return time

    def _mean_confidence_interval(self, data):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + self.confidence) / 2., n - 1)
        return m, m - h, m + h, h

    def _build_dfs(self, path_att, path_rank, path_rank_gh, team):
        s = requests.get(path_att).content
        df_att = pd.read_csv(io.StringIO(s.decode('ISO-8859-1')), sep=',', encoding="ISO-8859-1",
                             usecols=['Date', 'Team', 'Time', 'Opponent', 'Site', 'TV', 'Attendance', 'Current Wins',
                                      'Current Losses', 'Stadium Capacity', 'Fill Rate', 'New Coach', 'Tailgating',
                                      'PRCP',
                                      'SNOW', 'SNWD', 'TMAX', 'TMIN', 'Conference'])

        df_att['Team'] = df_att['Team'].apply(self._name_filter)
        df_att['Opponent'] = df_att['Opponent'].apply(self._name_filter)
        df_att['Year'] = df_att['Date'].apply(lambda x: x.split('/')[2].strip())
        df_att['Day'] = df_att['Date'].apply(lambda x: x.split('/')[1].strip())
        df_att['Month'] = df_att['Date'].apply(lambda x: x.split('/')[0].strip())
        df_att['Time'] = df_att['Time'].apply(self._convert_time)
        df_att = df_att.dropna()

        t = requests.get(path_rank_gh).text
        _all_rank_files = re.findall('cf\d\d\d\d.csv', t)
        _all_rank_files = [path_rank+file for file in _all_rank_files]
        all_rank_dfs = [pd.read_csv(f, sep=',', encoding="ISO-8859-1", names=['Sport Year', 'Team ID', 'Team Name',
                                                                              'Ranking System ID',
                                                                              'Ranking System Name', 'Date',
                                                                              'Ordinal Ranking'])
                        for f in _all_rank_files]
        self.df_rank = pd.concat(all_rank_dfs).drop(['Team ID', 'Ranking System Name', 'Date'], axis=1)
        self.df_rank['Sport Year'] = self.df_rank['Sport Year'].apply(lambda x: x.replace('cf', '').strip())
        self.df_rank['Team Name'] = self.df_rank['Team Name'].apply(lambda x: x.strip())
        self.df_rank['Ranking System ID'] = self.df_rank['Ranking System ID'].apply(lambda x: x.strip())
        self.df_rank = self.df_rank.groupby(['Team Name', 'Sport Year']).mean()

        df_att['Rank'] = df_att.apply(self._get_rank, axis=1)
        df_att['Opponent Rank'] = df_att.apply(self._get_opponent_rank, axis=1)
        df_att = df_att.dropna().drop(['Date'], axis=1).set_index(['Team', 'Opponent'])
        df_att['Rank Delta'] = df_att.apply(self._get_rank_delta, axis=1)
        df_att['Rank Average'] = df_att.apply(self._get_rank_average, axis=1)

        z = np.abs(stats.zscore(df_att['Fill Rate']))
        df_att = df_att[(z < 3)]  # throw out fill rates greater than 3 standard deviations above the mean
        df_att_numeric = df_att.copy().drop(
            ['Site', 'TV', 'SNOW', 'SNWD', 'New Coach', 'Tailgating', 'Conference', 'Rank Average'], axis=1)
        df_att_categorical = df_att.copy().drop(
            ['Time', 'Attendance', 'Current Wins', 'Current Losses', 'Stadium Capacity'
                , 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'Year', 'Day', 'Month',
             'Rank', 'Opponent Rank', 'Rank Delta', 'Rank Average'], axis=1)

        df_att_numeric = df_att_numeric.dropna().sort_index()

        if team != 'National':
            df_att_numeric_local = df_att_numeric.sort_index().loc[team]
            df_att_categorical_local = df_att_categorical.sort_index().loc[team]
            return df_att_numeric, df_att_categorical, df_att_numeric_local, df_att_categorical_local
        else:
            return df_att_numeric, df_att_categorical, None, None


if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    att_path_raw_github = "https://raw.githubusercontent.com/yaonuma/College-Football-Stadium-Fill-Rates/master/College%20Football%20Attendance/CFBeattendance.csv"
    rankings_path_github = 'https://github.com/yaonuma/College-Football-Stadium-Fill-Rates/tree/master/College%20Sports%20Rankings'
    rankings_path_raw_github = 'https://raw.githubusercontent.com/yaonuma/College-Football-Stadium-Fill-Rates/master/College%20Sports%20Rankings/'
    team = 'Michigan St'
    df_att_numeric, df_att_categorical, df_att_numeric_local, df_att_categorical_local = \
        CFFillRate(att_path_raw_github, rankings_path_github, rankings_path_raw_github, team).dfs

    # **********************************************************************************************************************

    df = df_att_numeric.copy()
    vars = df.columns
    rhovalues = []
    pvalues = []

    for i in range(len(vars)):
        print(vars[i], stats.spearmanr(df[vars[i]], df['Fill Rate']))
        stats_ = stats.spearmanr(df[vars[i]], df['Fill Rate'])
        if stats_[1] <= 0.001:
            rhovalues.append((vars[i], stats_))
    rhovalues = sorted(rhovalues.copy(), key=lambda x: abs(x[1][0]))
    top_rhos = rhovalues[-14:-2]

    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(rotation=90)
    plt.show()

    # print("Top Nation Variables")
    # for i in range(len(top_rhos)):
    #     print(top_rhos[i])
    f, axes = plt.subplots(2, 2)
    f.suptitle("Top Four Variables (SR) - National Average", fontsize=12, y=0.95)
    sns.regplot(x=top_rhos[-1][0], y='Fill Rate', data=df, order=1,
                scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
                line_kws={"color": "red"}, ax=axes[0, 0])
    sns.regplot(x=top_rhos[-2][0], y='Fill Rate', data=df, order=1,
                scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
                line_kws={"color": "red"}, ax=axes[0, 1])
    sns.regplot(x=top_rhos[-3][0], y='Fill Rate', data=df, order=1,
                scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
                line_kws={"color": "red"}, ax=axes[1, 0])
    sns.regplot(x=top_rhos[-4][0], y='Fill Rate', data=df, order=1,
                scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
                line_kws={"color": "red"}, ax=axes[1, 1])

    axes[0, 0].set_title("SR = " + str(round(float(top_rhos[-1][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos[-1][1][1]), 4)), loc='left', fontsize=10)
    axes[0, 1].set_title("SR = " + str(round(float(top_rhos[-2][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos[-2][1][1]), 4)), loc='right', fontsize=10)
    axes[1, 0].set_title("SR = " + str(round(float(top_rhos[-3][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos[-3][1][1]), 4)), loc='left', fontsize=10)
    axes[1, 1].set_title("SR = " + str(round(float(top_rhos[-4][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos[-4][1][1]), 4)), loc='right', fontsize=10)
    # f.tight_layout(pad=0.125)
    plt.show()
    # **********************************************************************************************************************

    # df = df_att_numeric_local.copy()
    # z_prcp = np.abs(stats.zscore(df['PRCP']))
    # df = df[(z_prcp < 3)]  # throw out fill rates greater than 3 standard deviations above the mean
    # vars = df.columns
    # rhovalues = []
    # for i in range(len(vars)):
    #     # rhovalue = stats.spearmanr(df[vars[i]], df['Fill Rate'])[0]
    #     stats_ = stats.spearmanr(df[vars[i]], df['Fill Rate'])
    #     if stats_[1] <= 0.001:
    #         rhovalues.append((vars[i], stats_))
    # rhovalues = sorted(rhovalues.copy(), key=lambda x: abs(x[1][0]))
    # top_rhos_local = rhovalues[-14:-2]
    #
    # f, axes = plt.subplots(2, 2)
    # f.suptitle("Select Variables (SR) - " + team, fontsize=12, y=0.95)
    # sns.regplot(x='Opponent Rank', y='Fill Rate', data=df, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[0, 0])
    # sns.regplot(x='Rank Delta', y='Fill Rate', data=df, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[0, 1])
    # sns.regplot(x='PRCP', y='Fill Rate', data=df, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[1, 0])
    # sns.regplot(x='TMAX', y='Fill Rate', data=df, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[1, 1])
    #
    # axes[0, 0].set_title("SR = " + str(round(float(top_rhos_local[-3][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-3][1][1]), 4)), loc='left', fontsize=10)
    # axes[0, 1].set_title("SR = " + str(round(float(top_rhos_local[-4][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-4][1][1]), 4)), loc='right', fontsize=10)
    # axes[1, 0].set_title("SR = " + str(round(float(top_rhos_local[-6][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-6][1][1]), 4)), loc='left', fontsize=10)
    # axes[1, 1].set_title("SR = " + str(round(float(top_rhos_local[-5][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-5][1][1]), 4)), loc='right', fontsize=10)
    # # f.tight_layout(pad=0.125)
    # # **********************************************************************************************************************
    #
    # df = df_att_numeric.copy()
    # z_prcp = np.abs(stats.zscore(df['PRCP']))
    # df = df[(z_prcp < 3)]  # throw out fill rates greater than 3 standard deviations above the mean
    #
    # f, axes = plt.subplots(2, 2)
    # f.suptitle("Select Variables (SR) - " + team + " vs. National Average", fontsize=12, y=0.95)
    # sns.regplot(x='Opponent Rank', y='Fill Rate', data=df, order=1,
    #             scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
    #             line_kws={"color": "red"}, ax=axes[0, 0], label="Nat'l")
    # sns.regplot(x='Rank Delta', y='Fill Rate', data=df, order=1,
    #             scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
    #             line_kws={"color": "red"}, ax=axes[0, 1], label="Nat'l")
    # sns.regplot(x='PRCP', y='Fill Rate', data=df, order=1, scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
    #             line_kws={"color": "red"}, ax=axes[1, 0], label="Nat'l")
    # sns.regplot(x='TMAX', y='Fill Rate', data=df, order=1, scatter_kws={"color": "lightcoral", "alpha": 0.25, "s": 10},
    #             line_kws={"color": "red"}, ax=axes[1, 1], label="Nat'l")
    #
    #
    # df_local = df_att_numeric_local.copy()
    #
    #
    # z_prcp = np.abs(stats.zscore(df_local['PRCP']))
    # df_local = df_local[(z_prcp < 3)]  # throw out fill rates greater than 3 standard deviations above the mean
    #
    # sns.regplot(x='Opponent Rank', y='Fill Rate', data=df_local, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[0, 0], label=team)
    # sns.regplot(x='Rank Delta', y='Fill Rate', data=df_local, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[0, 1], label=team)
    # sns.regplot(x='PRCP', y='Fill Rate', data=df_local, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[1, 0], label=team)
    # sns.regplot(x='TMAX', y='Fill Rate', data=df_local, order=1, scatter_kws={"color": "lightgreen"},
    #             line_kws={"color": "green"}, ax=axes[1, 1], label=team)
    #
    # axes[0, 0].set_title("SR Nat'l = " + str(round(float(top_rhos[-4][1][0]), 2)) + '  |  SR ' + team + '= ' + str(round(float(top_rhos_local[-3][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-3][1][1]), 4)), loc = 'left', fontsize=10)
    # axes[0, 1].set_title("SR Nat'l = " + str(round(float(top_rhos[-6][1][0]), 2)) + '  |  SR ' + team + '= ' + str(round(float(top_rhos_local[-4][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-4][1][1]), 4)), loc = 'right', fontsize=10)
    # axes[1, 0].set_title("SR Nat'l = " + str(round(float(top_rhos[-11][1][0]), 2)) + '  |  SR ' + team + '= ' + str(round(float(top_rhos_local[-6][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-6][1][1]), 4)), loc = 'left', fontsize=10)
    # axes[1, 1].set_title("SR Nat'l = " + str(round(float(top_rhos[-10][1][0]), 2)) + '  |  SR ' + team + '= ' + str(round(float(top_rhos_local[-5][1][0]), 2)) + ', '+"p-value = " + str(round(float(top_rhos_local[-5][1][1]), 4)), loc = 'right', fontsize=10)
    # # f.tight_layout(pad=0.125)
    # axes[0, 0].legend(loc='upper left')
    # axes[0, 1].legend(loc='upper right')
    # axes[1, 0].legend(loc='upper left')
    # axes[1, 1].legend(loc='upper right')
    # plt.show()


    # **********************************************************************************************************************
    # cateforical plots are not valid without error bars
    # **********************************************************************************************************************

    #
    # print(len(list(set(df_att_categorical['TV'].values))), ' Stations')
    # dfbar_tv = df_att_categorical.copy()
    # dfbar_tv = dfbar_tv.groupby('TV').mean().reset_index()
    # # yerr = [stat[3] for stat in [_mean_confidence_interval(data) for data in dfbar_tv['TV'].values]]
    #
    # df = dfbar_tv.sort_values(by=['Fill Rate'], ascending=False)
    # df['TV'] = df['TV'].apply(lambda x: 'Yes' if x.strip() != 'Not on TV' else 'No')
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("Aired On TV - National Average", fontsize=12)
    # ax = sns.barplot(x='TV', y='Fill Rate', data=df, capsize=.2, palette="Blues_d")
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # # **********************************************************************************************************************
    #
    # dfbar_team = df_att_categorical.copy().reset_index().groupby('Team').mean().reset_index().sort_values(
    #     by=['Fill Rate'], ascending=False)
    # print(len(list(set(dfbar_team['Team'].values))), ' Teams')
    # df = dfbar_team
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("Home Game Fill Rates - National", fontsize=12)
    # ax = sns.barplot(x='Team', y='Fill Rate', data=df, capsize=1, palette="Blues_d")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # # **********************************************************************************************************************
    #
    # dfbar_opponent = df_att_categorical.copy().reset_index().groupby('Opponent').mean().reset_index().sort_values(
    #     by=['Fill Rate'], ascending=False)
    # print(len(list(set(dfbar_opponent['Opponent'].values))), ' Opponents')
    # df = dfbar_opponent
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("Opponent Fill Rates - National", fontsize=12)
    # ax = sns.barplot(x='Opponent', y='Fill Rate', data=df, capsize=1, palette="Blues_d")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # # **********************************************************************************************************************
    #
    # dfbar_newcoach = df_att_categorical.copy().reset_index().groupby('New Coach').mean().reset_index().sort_values(
    #     by=['Fill Rate'], ascending=False)
    # print(len(list(set(dfbar_newcoach['New Coach'].values))), ' New Coaches (binary)')
    # df = dfbar_newcoach
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("New Coach Fill Rates - National Average", fontsize=12)
    # ax = sns.barplot(x='New Coach', y='Fill Rate', data=df, capsize=1, palette="Blues_d")
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # # **********************************************************************************************************************
    #
    # dfbar_tailgating = df_att_categorical.copy().reset_index().groupby('Tailgating').mean().reset_index().sort_values(
    #     by=['Fill Rate'], ascending=False)
    # print(len(list(set(dfbar_tailgating['Tailgating'].values))), ' Tailgate Categories')
    # df = dfbar_tailgating
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("Tailgating Fill Rates - National", fontsize=12)
    # ax = sns.barplot(x='Tailgating', y='Fill Rate', data=df, capsize=1, palette="Blues_d")
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # # **********************************************************************************************************************
    #
    # dfbar_conference = df_att_categorical.copy().reset_index().groupby('Conference').mean().reset_index().sort_values(
    #     by=['Fill Rate'], ascending=False)
    # print(len(list(set(dfbar_conference['Conference'].values))), ' Conferences')
    # df = dfbar_conference
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("Conference Fill Rates - National", fontsize=12)
    # ax = sns.barplot(x='Conference', y='Fill Rate', data=df, capsize=1, palette="Blues_d")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # # **********************************************************************************************************************
    #
    # # print(df_att_categorical_local.head())
    # dfbar_opponent = df_att_categorical_local.copy().reset_index().groupby(
    #     'Opponent').mean().reset_index().sort_values(by=['Fill Rate'], ascending=False)
    # # print(dfbar_opponent.head(100))
    # num_categories = len(list(set(dfbar_opponent['Opponent'].values)))
    # print(len(list(set(dfbar_opponent['Opponent'].values))), team+'  Opponents')
    # df = dfbar_opponent
    # f, axes = plt.subplots(1, 1)
    # f.suptitle("Opponent Fill Rates - " + team, fontsize=12)
    # ax = sns.barplot(x='Opponent', y='Fill Rate', data=df, palette="Blues_d")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #
    # plt.show()
