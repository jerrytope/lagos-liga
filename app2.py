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
