import streamlit as st
import pandas as pd
import html
from funcs import *

power_ratings = pd.read_csv('data/power_2025.csv')
rpi = pd.read_csv('data/rpi_2025.csv')

teams = power_ratings['team'].unique()
teams = [html.unescape(team) for team in teams]

# Ensure the same length for Opponent and Game Type
def make_schedule():
    base = [
        ('', 'Buzz Classic'), ('', 'Buzz Classic'), ('', 'Buzz Classic'),
        ('', 'Buzz Classic'), ('', 'Buzz Classic'), ('', 'Buzz Classic'),
        ('Florida', 'UF Tournament'), ('Florida', 'UF Tournament'), ('FIU', 'UF Tournament'), 
        ('FIU', 'UF Tournament'), ('Marshall', 'UF Tournament'), ('Marshall', 'UF Tournament'), 
        ('', 'I75'), ('', 'I75'), ('', 'I75'), ('', 'I75'), ('', 'I75'), ('', 'I75'), 
        ('Mercer', 'Midweek'),
        ('Notre Dame', 'Conference'), ('Notre Dame', 'Conference'), ('Notre Dame', 'Conference'),
        ('', 'Midweek'),
        ('Clemson', 'Conference'), ('Clemson', 'Conference'), ('Clemson', 'Conference'),
        ('', 'Midweek'),
        ('Virginia', 'Conference'), ('Virginia', 'Conference'), ('Virginia', 'Conference'),
        ('Georgia', 'Midweek'),
        ('Duke', 'Conference'), ('Duke', 'Conference'), ('Duke', 'Conference'),
        ('', 'Spring Break'), ('', 'Spring Break'), ('', 'Spring Break'),
        ('Missouri', 'Midweek'),
        ('Louisville', 'Conference'), ('Louisville', 'Conference'), ('Louisville', 'Conference'),
        ('', 'Midweek'),
        ('Boston College', 'Conference'), ('Boston College', 'Conference'), ('Boston College', 'Conference'),
        ('Jacksonville St.', 'Midweek'),
        ('', 'Bye'), ('', 'Bye'), ('', 'Bye'), 
        ('Ga. Southern', 'Midweek'),
        ('Virginia Tech', 'Conference'), ('Virginia Tech', 'Conference'), ('Virginia Tech', 'Conference'),
        ('Florida St.', 'Conference'), ('Florida St.', 'Conference'), ('Florida St.', 'Conference')
    ]
    return pd.DataFrame(base, columns=['Opponent', 'Game Type'])

established_schedule_df = make_schedule()

# Session state to store the schedule
if 'schedule' not in st.session_state or not isinstance(st.session_state.schedule, pd.DataFrame):
    st.session_state.schedule = established_schedule_df.copy()

st.title("Schedule Builder")

# Avoid the disappearing text bug by storing a local editable copy
editable_schedule = st.session_state.schedule.copy()

tabs = st.tabs(["Schedule Editor", "Season Simulation"])

with tabs[0]:

    with st.expander("How to Use", expanded=False):
        st.markdown("""
        **Step-by-step Guide:**

        1. **Team Name Input**: Use the dropdown to get the correct team name format.
        2. **Edit Schedule**: Modify opponent names and game types directly in the table. Add or remove rows as needed.
        3. **Validation**: Opponents not recognized will trigger a warning. Ensure names match the dropdown exactly.
        4. **Download Schedule**: Use the button to export your edited schedule as a CSV.

        Once your schedule is complete, head to the **Season Simulation** tab to simulate outcomes and view RPI estimates.
        """)

    team_names = st.selectbox(label='Proper Team Names for Input', options=teams)

    uploaded_file = st.file_uploader("Upload Schedule CSV (must include 'Opponent' and 'Game Type' columns)", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if {'Opponent', 'Game Type'}.issubset(uploaded_df.columns):
                st.session_state.schedule = uploaded_df
                editable_schedule = uploaded_df
                st.success("Schedule uploaded successfully!")
            else:
                st.error("CSV must contain 'Opponent' and 'Game Type' columns.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    def update_schedule():
        st.session_state.schedule = st.session_state.schedule_editor.copy()

    with st.container():
        edited_df = st.data_editor(
            editable_schedule,
            num_rows="dynamic",
            use_container_width=True,
            height=1000,
            key="schedule_editor",
            on_change=update_schedule
        )

    # Only update session state if something changed
    if not edited_df.equals(st.session_state.schedule):
        st.session_state.schedule = edited_df

    merged_df = st.session_state.schedule.merge(
        rpi[['team_name', 'rpi_coef']],
        left_on='Opponent',
        right_on='team_name',
        how='left'
    )   
    merged_df = merged_df.rename(columns={'rpi_coef': 'Opponent RPI'}).drop(columns=['team_name'])

    # Show warning if any opponent isn't recognized and not blank
    invalid_opponents = merged_df[
        (~merged_df['Opponent'].isin(teams)) &
        merged_df['Opponent'].notna() &
        (merged_df['Opponent'].str.strip() != '')
    ]['Opponent'].unique()

    if len(invalid_opponents) > 0:
        st.warning(f"These opponent names are not recognized: {', '.join(invalid_opponents)}")

    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "schedule.csv", "text/csv")

with tabs[1]:

    with st.expander("Config Settings", expanded=False):
        default_gt_power = float(power_ratings[power_ratings['team'] == 'Georgia Tech']['power_rating'].values[0])
        gt_power_rating = st.number_input(
            label="Georgia Tech Power Rating",
            min_value=0.0,
            max_value=100.0,
            value=default_gt_power,
            step=0.1,
            format="%.3f"
        )

        custom_power_df = power_ratings.copy()
        custom_power_df.loc[custom_power_df['team'] == 'Georgia Tech', 'power_rating'] = gt_power_rating

        # Compute rank (1 = highest rating)
        custom_power_df['rank'] = custom_power_df['power_rating'].rank(ascending=False, method='min')
        gt_rank = int(custom_power_df[custom_power_df['team'] == 'Georgia Tech']['rank'].values[0])
        total_teams = len(custom_power_df)

        st.markdown(f"**Ranking with this Power Rating:** {gt_rank} out of {total_teams}")

        st.session_state.parity = st.number_input(
            label="Parity (0 = chaotic, 1 = chalk)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f"
        )
    
    # Merge with RPI values
    try:
        merged_df = st.session_state.schedule.merge(
            rpi[['team_name', 'rpi_coef']],
            left_on='Opponent',
            right_on='team_name',
            how='left'
        )
        merged_df = merged_df.rename(columns={'rpi_coef': 'Opponent RPI'}).drop(columns=['team_name'])
    except Exception:
        merged_df = st.session_state.schedule

    n = st.number_input(label='Num. of Simulations', min_value=1, step=1, format="%d")

    st.subheader("Simulation Results")

    if st.button("Run Simulations") and n > 0:
        full_schedule = merged_df['Opponent'].dropna()
        full_schedule = full_schedule[full_schedule.str.strip() != ''].tolist()

        opponent_counts = pd.Series(full_schedule).value_counts().to_dict()
        win_counts = {opp: 0 for opp in opponent_counts}
        rpi_vals = []
        rpi_ranks = []
        total_wins = 0
        total_games = len(full_schedule)

        progress_bar = st.progress(0, text='Simulating Seasons...')

        for i in range(n):
            sim_df = simulate_season(full_schedule)

            wins = sim_df['win'].sum()
            total_wins += wins

            win_by_opp = sim_df[sim_df['win']].groupby('opponent').size()
            for opp, wins in win_by_opp.items():
                win_counts[opp] += wins

            rpi_val, rpi_rank = calculate_rpi(sim_df)
            rpi_vals.append(rpi_val)
            rpi_ranks.append(rpi_rank)

            progress_bar.progress((i + 1) / n, text=f"Simulating seasons... ({i + 1}/{n})")


        merged_df = merged_df.copy()
        merged_df['Win%'] = merged_df['Opponent'].apply(
            lambda x: f"{(win_counts.get(x, 0) / (n * opponent_counts.get(x, 1)) * 100):.1f}%"
            if x in opponent_counts else ""
        )

        avg_rpi = np.mean(rpi_vals)
        avg_rank = np.mean(rpi_ranks)
        avg_wins = total_wins / n
        avg_losses = total_games - avg_wins

        st.dataframe(merged_df[['Opponent', 'Game Type', 'Opponent RPI', 'Win%']], use_container_width=True)

        st.markdown(f"**Average W–L Record:** {int(round(avg_wins))}–{int(round(avg_losses))}")
        st.markdown(f"**Average RPI:** {avg_rpi:.4f}")
        st.markdown(f"**Average RPI Rank:** {avg_rank:.1f}")