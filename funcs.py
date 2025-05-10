import pandas as pd
import random
import numpy as np

power_ratings = pd.read_csv('data/power_2025.csv')
rpi_df = pd.read_csv('data/rpi_2025.csv')
win_prob = pd.read_csv('data/win_prob.csv').drop('Unnamed: 0', axis=1)
win_perc_df = pd.read_csv('data/win_perc.csv').drop('Unnamed: 0', axis=1)
opponent_win_perc_df = pd.read_csv('data/opponent_win_perc.csv').drop('Unnamed: 0', axis=1)

# simulate a single game between two teams using win perc matrix
# inputs: team names
# output: winning team name
def simulate_game(team1, team2, parity=0.5):
    matchup = win_prob[(win_prob['team1'] == team1) & (win_prob['team2'] == team2)]

    win_probability = matchup['team1_winperc'].iloc[0]

    # Adjust win probability based on parity
    # 0.5 = unchanged; 0 = flat 50/50; 1 = extreme (closer to 0 or 1)
    adjusted_prob = 0.5 + (win_probability - 0.5) * parity

    if random.random() < adjusted_prob:
        return team1
    else:
        return team2
    

# given hypothetical schedule, simulate a single time
# input: input_schedule (list)
# output: schedule_df
## cols = opponent, win (bool)
def simulate_season(input_schedule):

    schedule_df = pd.DataFrame()

    for opponent in input_schedule:

        if (opponent == '') or (opponent is None):
            next

        result = simulate_game('Georgia Tech', opponent)

        if result == 'Georgia Tech':
            win = True
        else:
            win = False

        schedule_df = pd.concat([schedule_df, pd.DataFrame({"opponent": opponent, "win": win}, index=[0])])

    return schedule_df
    
# given a teams schedule with wins and losses, calculate RPI
# input: schedule_df
## cols = opponent, win (bool)
# output: rpi (float)
def calculate_rpi(schedule_df):
    # calculate win percentage
    win_perc = schedule_df['win'].mean()

    schedule_w_opp_win_perc = pd.merge(schedule_df, win_perc_df, left_on='opponent', right_on='team_name', how='left').drop('team_name', axis=1)
    
    opponent_win_perc = schedule_w_opp_win_perc['win_perc'].mean()

    schedule_w_opp_opp_win_perc = pd.merge(schedule_df, opponent_win_perc_df, left_on='opponent', right_on='team_name', how='left').drop('team_name', axis=1)

    opponent_opponent_win_perc = schedule_w_opp_opp_win_perc['opponent_win_perc'].mean()

    rpi = (win_perc * .25) + (opponent_win_perc * .5) + (opponent_opponent_win_perc * .25)

    all_rpi_vals = rpi_df['rpi_coef'].tolist() + [rpi]
    rpi_rank = sorted(all_rpi_vals, reverse=True).index(rpi) + 1

    return rpi, rpi_rank


# ex_input_schedule = ['Georgia St.', 'Alabama', 'Florida']

# schedule_df = simulate_season(ex_input_schedule)

# rpi, rpi_rank = calculate_rpi(schedule_df)

# print(rpi, rpi_rank)
