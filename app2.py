import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from PIL import Image, ImageEnhance

# Constants
DOCUMENT_ID = '1CWOu-S9NtRn49wXP_7as2sknb82Uvj5YUWy-Ot4g3wE'
LOGO_PATH = "logos/logo.png"

# Cached function for fetching data
@st.cache_data
def fetch_data(sheet_name: str) -> pd.DataFrame:
    """Fetch data from Google Sheets."""
    url = f'https://docs.google.com/spreadsheets/d/{DOCUMENT_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url, index_col=0)

# Helper Functions
def apply_logo(fig, logo_path: str, opacity: float = 0.2):
    """Overlay a logo on a matplotlib figure."""
    logo = Image.open(logo_path).resize((300, 300), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Brightness(logo)
    logo = enhancer.enhance(opacity)
    fig.figimage(logo, xo=fig.bbox.x0, yo=fig.bbox.y0, origin='upper')
    return fig

def filter_team_games(data: pd.DataFrame, team: str) -> pd.DataFrame:
    """Filter games involving a specific team."""
    return data[(data['home_team_name'] == team) | (data['away_team_name'] == team)]

def calculate_head_to_head_totals(team_data: pd.DataFrame, team1: str, team2: str):
    """Calculate head-to-head statistics for two teams."""
    head_to_head_data = team_data[((team_data['home_team_name'] == team1) & (team_data['away_team_name'] == team2)) |
                                  ((team_data['home_team_name'] == team2) & (team_data['away_team_name'] == team1))]

    def compute_totals(team: str, column: str):
        home_goals = head_to_head_data[head_to_head_data['home_team_name'] == team][column].sum()
        away_goals = head_to_head_data[head_to_head_data['away_team_name'] == team][column].sum()
        return home_goals + away_goals

    goals_scored_team1 = compute_totals(team1, 'home_team_score')
    goals_scored_team2 = compute_totals(team2, 'home_team_score')
    goals_conceded_team1 = compute_totals(team1, 'away_team_score')
    goals_conceded_team2 = compute_totals(team2, 'away_team_score')

    return (goals_scored_team1, goals_conceded_team1), (goals_scored_team2, goals_conceded_team2)

def create_bar_chart(data, title: str, x_col: str, y_col: str):
    """Create a horizontal bar chart using seaborn."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=data, palette="Reds")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    return plt.gcf()

# Main Streamlit App
# def main():
#     st.title("FIFA U20 Women's World Cup")

#     # Data Loading
#     matches = fetch_data('matches')
#     team_stats = fetch_data('TEAM_STATS')
#     player_stats = fetch_data('PLAYER_STATS')

#     # Team Selection
#     all_teams = sorted(list(set(matches['home_team_name']).union(set(matches['away_team_name']))))
#     selected_teams = st.multiselect("Select Teams", all_teams, default=None)
#     if len(selected_teams) != 2:
#         st.warning("Please select exactly two teams.")
#         return
#     team1, team2 = selected_teams

#     # Head-to-Head Analysis
#     st.subheader("Head-to-Head Analysis")
#     (team1_stats, team1_conceded), (team2_stats, team2_conceded) = calculate_head_to_head_totals(matches, team1, team2)
#     st.write(f"{team1}: Goals Scored - {team1_stats}, Goals Conceded - {team1_conceded}")
#     st.write(f"{team2}: Goals Scored - {team2_stats}, Goals Conceded - {team2_conceded}")

def main():
    st.title("FIFA U20 Women's World Cup")

    # Data Loading
    matches = fetch_data('matches')
    team_stats = fetch_data('TEAM_STATS')
    player_stats = fetch_data('PLAYER_STATS')

    # Clean Team Names
    matches['home_team_name'] = matches['home_team_name'].fillna('').astype(str)
    matches['away_team_name'] = matches['away_team_name'].fillna('').astype(str)

    # Team Selection
    all_teams = sorted(list(set(matches['home_team_name']).union(set(matches['away_team_name']))))
    selected_teams = st.multiselect("Select Teams", all_teams, default=None)

    if len(selected_teams) != 2:
        st.warning("Please select exactly two teams.")
        return

    team1, team2 = selected_teams

    # Head-to-Head Analysis
    st.subheader("Head-to-Head Analysis")
    (team1_stats, team1_conceded), (team2_stats, team2_conceded) = calculate_head_to_head_totals(matches, team1, team2)
    st.write(f"{team1}: Goals Scored - {team1_stats}, Goals Conceded - {team1_conceded}")
    st.write(f"{team2}: Goals Scored - {team2_stats}, Goals Conceded - {team2_conceded}")

    # Further analysis and visualizations...


    # Goals Distribution
    st.subheader("Goals Distribution by Stage")
    goals_distribution = matches.groupby('stage').agg(
        home_goals=('home_team_score', 'sum'),
        away_goals=('away_team_score', 'sum')
    ).reset_index()
    goals_distribution['total_goals'] = goals_distribution['home_goals'] + goals_distribution['away_goals']
    fig = create_bar_chart(goals_distribution, "Goals Distribution by Stage", 'total_goals', 'stage')
    st.pyplot(fig)

    # Player Stats Comparison
    st.subheader("Player Stats Comparison")
    team1_players = player_stats[player_stats['Team'] == team1]['Player'].unique()
    team2_players = player_stats[player_stats['Team'] == team2]['Player'].unique()
    player1 = st.selectbox(f"Select Player from {team1}", team1_players)
    player2 = st.selectbox(f"Select Player from {team2}", team2_players)

    # Radar Chart
    stats = ['Goals Scored', 'Assists', 'Shots On Target']
    player1_data = player_stats[player_stats['Player'] == player1][stats].sum()
    player2_data = player_stats[player_stats['Player'] == player2][stats].sum()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player1_data,
        theta=stats,
        fill='toself',
        name=player1
    ))
    fig.add_trace(go.Scatterpolar(
        r=player2_data,
        theta=stats,
        fill='toself',
        name=player2
    ))
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
















# data = fetch_data('matches')


# def generate_result_string(last_n_team_games):
#     return ''.join(last_n_team_games['result'].values)
# # Step 10: Data Visualization functions
# def filter_team_games(data, team):
#     return data[(data['home_team_name'] == team) | (data['away_team_name'] == team)]

# def get_last_n_games(team_games, n=3):
#     # Filter out games that haven't been played yet (missing goals)
#     completed_games = team_games.dropna(subset=['home_team_score', 'away_team_score'])
#     return completed_games.tail(n)

# def determine_result(row, team):
#     if team == row['home_team_name']:
#         if row['home_team_score'] > row['away_team_score']:
#             return 'W'
#         elif row['home_team_score'] < row['away_team_score']:
#             return 'L'
#         else:
#             return 'D'
#     elif team == row['away_team_name']:
#         if row['away_team_score'] > row['home_team_score']:
#             return 'W'
#         elif row['away_team_score'] < row['home_team_score']:
#             return 'L'
#         else:
#             return 'D'
#     else:
#         return None

# def display_last_n_games(last_n_team_games):
#     columns_to_display = ['home_team_name', 'away_team_name', 'home_team_score', 'away_team_score', 'result']
#     return last_n_team_games[columns_to_display]



# # Get unique team names
# unique_home_teams = data['home_team_name'].unique()
# unique_away_teams = data['away_team_name'].unique()

# # Remove home teams already present in the away teams list
# unique_away_teams = [team for team in unique_away_teams if team not in unique_home_teams]

# # Combine unique home and away team names
# all_unique_teams = list(unique_home_teams) + list(unique_away_teams)

# st.sidebar.header("Team Runs")

# # Use the multiselect widget with the combined unique team names
# selected_teams = st.sidebar.multiselect("Select Teams", all_unique_teams)

# # Ensure exactly two teams are selected

# if len(selected_teams) != 2:
#     st.sidebar.warning("Please select exactly two teams.")
# else:
#     # Unpack selected teams
#     team1, team2 = selected_teams

#     # Step 6: Filter the data based on the selected teams
#     team1_games = filter_team_games(data, team1)
#     team2_games = filter_team_games(data, team2)

#     # Get the last 5 games for each team
#     last_5_team1_games = get_last_n_games(team1_games)
#     last_5_team2_games = get_last_n_games(team2_games)

#     # Determine results for each team
#     last_5_team1_games['result'] = last_5_team1_games.apply(determine_result, axis=1, team=team1)
#     last_5_team2_games['result'] = last_5_team2_games.apply(determine_result, axis=1, team=team2)

#     # Generate result strings
#     team1_results_str = generate_result_string(last_5_team1_games)
#     team2_results_str = generate_result_string(last_5_team2_games)

#     # Display results and games
#     st.subheader(f"Last 5 games involving {team1}: {team1_results_str}")
#     st.write(display_last_n_games(last_5_team1_games))

#     st.subheader(f"Last 5 games involving {team2}: {team2_results_str}")
#     st.write(display_last_n_games(last_5_team2_games))



