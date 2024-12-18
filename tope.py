import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import mplsoccer
from mplsoccer.pitch import Pitch

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns
from matplotlib.lines import Line2D 
from matplotlib.lines import Line2D  # Import Line2D for custom legend elements
import io
from matplotlib import pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.colors as mcolors



document_id = '11GOW9_pzJmAAAlvWKFYdh7YCc0lF4U7w'

@st.cache_data
def fetch_data(sheet_name):
    url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url)

if st.button("Refresh Data"):
    st.cache_data.clear()



def download_plot(fig, file_name, mime="image/png", facecolor=None):
    """Downloads a Matplotlib figure as a PNG image.

    Args:
        fig (matplotlib.figure.Figure): The figure to download.
        file_name (str): The desired filename for the downloaded image.
        mime (str, optional): The MIME type of the image. Defaults to "image/png".
        facecolor (str, optional): The background color for the plot (used for heatmaps). Defaults to None.
    """
    buffer = io.BytesIO()
    if facecolor:
        # Set background color for heatmaps
        fig.set_facecolor(facecolor)
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    st.download_button(
        label="Download",
        data=buffer,
        file_name=file_name,
        mime=mime
    )


def plot_pass_map_for_player(ax, player_name, df_pass):
    # Filter the pass DataFrame for the selected player
    player_pass_df = df_pass[df_pass['player'] == player_name]

    # Create the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    ax.set_title(f'Pass Map for {player_name}', color='white', fontsize=10, pad=20)
    ax.invert_yaxis()

    # Initialize legend elements
    legend_elements = []

    # Plot each pass
    for _, row in player_pass_df.iterrows():
        if row['outcome'] == 'successful':
            ax.plot((row['x'], row['endX']), (row['y'], row['endY']), color='green')
            ax.scatter(row['x'], row['y'], color='green')
            if not any(el.get_label() == "Successful" for el in legend_elements):
                legend_elements.append(Line2D([0], [0], color='green', lw=2, label='Successful'))
        elif row['outcome'] == 'unsuccessful':
            ax.plot((row['x'], row['endX']), (row['y'], row['endY']), color='red')
            ax.scatter(row['x'], row['y'], color='red')
            if not any(el.get_label() == "Unsuccessful" for el in legend_elements):
                legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Unsuccessful'))

    # Add legend to the plot
    ax.legend(
        handles=legend_elements,
        loc='upper center',  # Move the legend upwards
        fontsize=8,
        facecolor='white',  # Change the background to white
        frameon=True,  # Ensure legend has a background
        edgecolor='black',  # Add a border for contrast
    )

import matplotlib.pyplot as plt
import io
import streamlit as st

def plot_pass_maps(player1, df_pass):
    # Create a single plot
    fig, ax = plt.subplots(figsize=(8, 7)) 

    # Set background color
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # Plot pass map for the player
    plot_pass_map_for_player(ax, player1, df_pass)

    # Display in Streamlit
    plt.xlim(0, 120)
    plt.ylim(0, 80)
    st.pyplot(fig)
    download_plot(fig, f"{player1}_pass_map.png")







df_pass = fetch_data("week_player")

st.sidebar.header("Pass Charts")

# Add "All games" option to the game selection
game_options = ["All games"] + df_pass["Game"].unique().tolist()
selected_pass_game = st.sidebar.selectbox("Select a game", game_options)

# Dropdown to select a team
if selected_pass_game == "All games":
    # If "All games" is selected, get unique teams across all games
    team_options = df_pass["Team"].unique()
else:
    # Filter by selected game and get unique teams
    filtered_pass_game = df_pass[df_pass["Game"] == selected_pass_game]
    team_options = filtered_pass_game["Team"].unique()

selected_pass_team = st.sidebar.selectbox("Select a team", options=team_options)

# Dropdown to select a player
if selected_pass_game == "All games":
    # Use all data for the selected team
    filtered_pass_team = df_pass[df_pass["Team"] == selected_pass_team]
else:
    # Filter by game and team
    filtered_pass_team = filtered_pass_game[filtered_pass_game["Team"] == selected_pass_team]

players = filtered_pass_team["player"].unique()
player1_pass = st.sidebar.selectbox("Select Player 1", options=players, key="player1", index=0)




st.header("Player Passes on Football Pitch")

# Plot pass maps using the filtered player data
plot_pass_maps(player1_pass, filtered_pass_team)




def create_team_heatmap(df_pass):
    st.sidebar.header("Select player for Heatmap")
    st.header("player Heatmap on Football Pitch")

    game_options = ["All Games"] + df_pass["Game"].unique().tolist()
    selected_game = st.sidebar.selectbox("Select Game:", game_options)

    # Step 1: Team selection
    player_heat = st.sidebar.selectbox("Select a team:", df_pass["player"].unique())

    # Step 2: Game selection (add a new filter)
    

    # Filter data for the selected team and game
    if selected_game == "All Games":
        team_data = df_pass[df_pass["player"] == player_heat]
    else:
        team_data = df_pass[(df_pass["player"] == player_heat) & (df_pass["Game"] == selected_game)]

    # Create the heatmap plot
    fig_heat, ax = plt.subplots(figsize=(6, 4))
    fig_heat.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')
    ax.set_title(f'{player_heat}  Heatmap - {selected_game}', color='white', fontsize=10, pad=20)

    # Create the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

    custom_colors = [(0, 'purple'), (0.5, 'yellow'), (1, 'red')]

    # Create custom colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("", custom_colors)

    # Generate heatmap
    sns.kdeplot(
    x=team_data['x'],
    y=team_data['y'],
    color='red',
    alpha=0.7,
    levels=10,
    cmap=custom_cmap,
    bw_adjust=0.3,  # Adjust this value to reduce spread (default is 1)
    shade=True,
    ax=ax
    )


    # Set pitch boundaries
    plt.xlim(0, 120)
    plt.ylim(0, 80)

    # Display the heatmap in Streamlit
    st.pyplot(fig_heat)
    download_plot(fig_heat, f"{player_heat}_heatmap.png", facecolor=pitch.pitch_color)
df_pass = fetch_data("week_player")
create_team_heatmap(df_pass)




df = fetch_data('PLAYER_STATS')

def create_radar_chart(df, player1, player2):
    import plotly.graph_objects as go

    # Extract metric columns
    metrics = df.columns[1:].tolist()
    
    # Sum data across all games for each player
    player1_data = df[df['Player'] == player1].iloc[:, 1:].sum().tolist()
    player2_data = df[df['Player'] == player2].iloc[:, 1:].sum().tolist()

    # Create traces
    trace1 = go.Scatterpolar(
        r=player1_data,
        theta=metrics,
        fill='toself',
        name=player1,
        marker=dict(color='blue')
    )

    trace2 = go.Scatterpolar(
        r=player2_data,
        theta=metrics,
        fill='toself',
        name=player2,
        marker=dict(color='red')
    )

    # Layout configuration
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(player1_data), max(player2_data)) * 1.2]  # Adjust range dynamically
            )
        ),
        showlegend=True
    )

    # Combine traces and create figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig


# Streamlit sidebar inputs
st.sidebar.header("Radar Chart Comparison")
team_options = df['Team'].unique()
team1 = st.sidebar.selectbox("Select Team 1", team_options, index=4)
team2 = st.sidebar.selectbox("Select Team 2", team_options, index=5)

# Filter teams and position
position_options_team = df['POS'].unique()
selected_position = st.sidebar.selectbox("Select Position", position_options_team)

# Define position-specific columns
if selected_position == 'GK':
    columns_to_use = ['Player', 'Clean sheet', 'Saves', 'Crosses Claimed', 'Clearances']
elif selected_position == 'FW':
    columns_to_use = ['Player', 'Goals', 'Shots on target', 'Assists', 'Chances Created', 'Pass complete', 'Key Passes']
elif selected_position == 'MF':
    columns_to_use = ['Player', 'Goals', 'Assists', 'Tackles', 'Interception', 'Chances Created', 'Key Passes', 'Fouls committed', 'Blocks', 'Goal Involvement']
elif selected_position == 'DF':
    columns_to_use = ['Player', 'Goals', 'Tackles', 'Interception', 'Blocks', 'Clearances', 'Fouls committed', 'Defensive Contribution', 'Error']

# Filter DataFrame based on teams, positions, and columns
df_team1_filtered = df[(df['Team'] == team1) & (df['POS'] == selected_position)]
df_team2_filtered = df[(df['Team'] == team2) & (df['POS'] == selected_position)]
filtered_df_radar = df[df['POS'] == selected_position][columns_to_use]

# Get unique players filtered by team and position
players_team1 = df_team1_filtered['Player'].unique().tolist()
players_team2 = df_team2_filtered['Player'].unique().tolist()

# Sidebar player selection
player1 = st.sidebar.selectbox(f"Select Player from {team1}", players_team1, index=0)
player2 = st.sidebar.selectbox(f"Select Player from {team2}", players_team2, index=0)

# Ensure two different players are selected
if player1 != player2:
    # Filter data for selected players and sum their metrics
    player1_value = filtered_df_radar[filtered_df_radar['Player'] == player1].groupby('Player').sum()
    player2_value = filtered_df_radar[filtered_df_radar['Player'] == player2].groupby('Player').sum()

    # Combine data for display
    merged_df = pd.concat([player1_value, player2_value], axis=0)
    st.write(merged_df)

    # Display radar chart
    st.header('Radar Chart Comparison')
    fig = create_radar_chart(filtered_df_radar, player1, player2)
    st.plotly_chart(fig)
else:
    st.sidebar.warning("Please select two different players.")
