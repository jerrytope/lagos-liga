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

# Google Sheets document ID
# https://docs.google.com/spreadsheets/d/11GOW9_pzJmAAAlvWKFYdh7YCc0lF4U7w/edit?gid=306235239#gid=306235239

# st. set_page_config(layout="wide")
st.set_page_config( page_title="Lagos Liga Analysis With MIAS")

# document_id = '1oCS-ubjn2FtmkHevCToSCfcgL6WpjgXA3qoGnsu8IWk'

# def fetch_data(sheet_name):
#     url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    # return pd.read_csv(url)
# document_id = '1XDRCoTNodcU28nk4HYxOzjBJzt_e5gEo'

# @st.cache_data
# def fetch_data(sheet_name):
#     # Construct the URL to fetch data as a CSV
#     url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
#     return pd.read_csv(url)

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

def create_plot(df, title):
    stats_column = 'stats'
    team1_column = df.iloc[0, 1]  # Home team
    team2_column = df.iloc[1, 1]  # Away team

    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame({
        stats_column: df.columns[2:],  # Extract stats from column names starting from the 3rd column
        team1_column: df.iloc[0, 2:],  # Extract data for Home Team
        team2_column: df.iloc[1, 2:]   # Extract data for Away Team
    })

    
    trace1 = go.Bar(
        y=plot_df[stats_column],
        x=-plot_df[team1_column],  # Make x-values negative for one team
        name=team1_column,
        orientation='h',
        marker=dict(color='blue'),
        text=[f"<b>{int(x)}</b>" for x in plot_df[team1_column]],  # Make text bold
        textposition='outside',  # Position labels outside the bars
        textfont=dict(color='black'),  # Set text color to black
        hoverinfo='x+text'  # Hover information
    )

    trace2 = go.Bar(
        y=plot_df[stats_column],
        x=plot_df[team2_column],
        name=team2_column,
        orientation='h',
        marker=dict(color='red'),
        text=[f"<b>{int(x)}</b>" for x in plot_df[team2_column]],  # Make text bold
        textposition='outside',  # Position labels outside the bars
        textfont=dict(color='black'),  # Set text color to black
        hoverinfo='x+text'  # Hover information 
    )


    

    layout = go.Layout(
        title=title,
        barmode='overlay',
        bargap=0.1,
        bargroupgap=0,
        xaxis=dict(
            title='Values',
            showgrid=False,  # Hide x-axis grid lines
            zeroline=True,
            showline=True,
            showticklabels=False  # Hide x-axis ticks
        ),
        yaxis=dict(
            title='Stats',
            showgrid=False,  # Hide y-axis grid lines
            showline=True,
            showticklabels=True,
            tickfont=dict(color='black'),  # Set stats values color to black
            categoryorder='array',  # Order by the values in the DataFrame
            categoryarray=list(plot_df[stats_column])[::-1]  # Use the order from the DataFrame and reverse it
        )
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Adding central line for visual reference
    fig.add_shape(type="line",
                  x0=0, y0=-0.5, x1=0, y1=len(plot_df[stats_column])-0.5,
                  line=dict(color="black", width=2))

    return fig


df2 = fetch_data('TEAM_STATS')
# df_team = fetch_data('TEAM_STATS')
df2 = df2[['match_id_new','team_name','Goals','Goal Attempts','Shots On Target','Shots Off Target', 'Fouls', 'Corners',  'Yellow Card', 'Red Card','Goalkeeper Saves', 'Crosses','Shootout goals']]

# df2 = df2[['match_id_new','team_name','Goals','Goal Attempts']]


match_id = df2['match_id_new'].unique()
st.sidebar.header("Match Statistics")
item = st.sidebar.selectbox("Select match ID", match_id)

match_id_filtered = df2[df2['match_id_new'] == item]


fig1 = create_plot(match_id_filtered, f'{match_id_filtered["team_name"].iloc[0]} vs {match_id_filtered["team_name"].iloc[1]}')
st.title("Lagos Liga Analysis With MIAS")
st.subheader("Match Day Team Stats")
st.plotly_chart(fig1)


df = fetch_data('PLAYER_STATS')
df = df.iloc[:, 5:60]

df_stats = df.iloc[:,3:55]
df_stats = df_stats.groupby('Player').sum(numeric_only=True).reset_index()


stats = ['Goals', 'Assists', 'Chances Created','Shots on target','Shots off target', 'Crosses', 'Fouls committed', 'Fouls drawn', 'Corner', 'Saves', 'Interception', 'Pass complete', 'Key Passes', 'Goal Involvement']




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



# def plot_pass_map_for_player(ax, player_name, df_pass):
#     # Filter the pass DataFrame for the selected player
#     player_pass_df = df_pass[df_pass['player'] == player_name]

#     # Create the pitch
#     pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
#     pitch.draw(ax=ax)
#     ax.set_title(f'Pass Map for {player_name}', color='white', fontsize=10, pad=20)
#     ax.invert_yaxis()

#     # Initialize legend elements
#     legend_elements = []

#     # Plot each pass
#     for _, row in player_pass_df.iterrows():
#         if row['outcome'] == 'successful':
#             ax.plot((row['x'], row['endX']), (row['y'], row['endY']), color='green')
#             ax.scatter(row['x'], row['y'], color='green')
#             if not any(el.get_label() == "Successful" for el in legend_elements):
#                 legend_elements.append(Line2D([0], [0], color='green', lw=2, label='Successful'))
#         elif row['outcome'] == 'unsuccessful':
#             ax.plot((row['x'], row['endX']), (row['y'], row['endY']), color='red')
#             ax.scatter(row['x'], row['y'], color='red')
#             if not any(el.get_label() == "Unsuccessful" for el in legend_elements):
#                 legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Unsuccessful'))

#     # Add legend to the plot
#     ax.legend(
#         handles=legend_elements,
#         loc='upper center',  # Move the legend upwards
#         fontsize=8,
#         facecolor='white',  # Change the background to white
#         frameon=True,  # Ensure legend has a background
#         edgecolor='black',  # Add a border for contrast
#     )

# import matplotlib.pyplot as plt
# import io
# import streamlit as st

# def plot_pass_maps(player1, df_pass):
#     # Create a single plot
#     fig, ax = plt.subplots(figsize=(8, 7)) 

#     # Set background color
#     fig.set_facecolor('#22312b')
#     ax.patch.set_facecolor('#22312b')

#     # Plot pass map for the player
#     plot_pass_map_for_player(ax, player1, df_pass)

#     # Display in Streamlit
#     st.pyplot(fig)
#     download_plot(fig, f"{player1}_pass_map.png")







# df_pass = fetch_data("Passes")

# st.sidebar.header("Pass Charts")

# # Add "All games" option to the game selection
# game_options = ["All games"] + df_pass["Game"].unique().tolist()
# selected_pass_game = st.sidebar.selectbox("Select a game", game_options)

# # Dropdown to select a team
# if selected_pass_game == "All games":
#     # If "All games" is selected, get unique teams across all games
#     team_options = df_pass["Team"].unique()
# else:
#     # Filter by selected game and get unique teams
#     filtered_pass_game = df_pass[df_pass["Game"] == selected_pass_game]
#     team_options = filtered_pass_game["Team"].unique()

# selected_pass_team = st.sidebar.selectbox("Select a team", options=team_options)

# # Dropdown to select a player
# if selected_pass_game == "All games":
#     # Use all data for the selected team
#     filtered_pass_team = df_pass[df_pass["Team"] == selected_pass_team]
# else:
#     # Filter by game and team
#     filtered_pass_team = filtered_pass_game[filtered_pass_game["Team"] == selected_pass_team]

# players = filtered_pass_team["player"].unique()
# player1_pass = st.sidebar.selectbox("Select Player 1", options=players, key="player1", index=0)




# st.header("Player Passes on Football Pitch")

# # Plot pass maps using the filtered player data
# plot_pass_maps(player1_pass, filtered_pass_team)



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch
import io
import streamlit as st

# # Function to fetch data
# def fetch_data(data_type):
#     # Replace with actual data fetching logic
#     pass

# # Function to download the plot
# def download_plot(fig, filename):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     buf.seek(0)
#     st.download_button(
#         label="Download Pass Map",
#         data=buf,
#         file_name=filename,
#         mime="image/png"
#     )

# Plot team pass map
def plot_pass_map_for_team(ax, team_name, team_data):
    # Create the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    ax.set_title(f'Pass Map for {team_name} - {selected_pass_game}', color='white', fontsize=10, pad=20)
    ax.invert_yaxis()

    # Initialize legend elements
    legend_elements = []

    # Plot each pass
    for _, row in team_data.iterrows():
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
        loc='upper center',
        fontsize=8,
        facecolor='white',
        frameon=True,
        edgecolor='black',
    )

# Main function to plot the map
def plot_pass_maps(team_name, team_data):
    # Create a single plot
    fig, ax = plt.subplots(figsize=(8, 7))

    # Set background color
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    # Plot pass map for the team
    plot_pass_map_for_team(ax, team_name, team_data)

    # Display in Streamlit
    st.pyplot(fig)
    download_plot(fig, f"{team_name}_pass_map.png")

# Load the pass data
df_pass = fetch_data("Passes")

# Streamlit UI
st.sidebar.header("Pass Charts")

# Add "All games" option to the game selection
game_options = ["All games"] + df_pass["Game"].unique().tolist()
selected_pass_game = st.sidebar.selectbox("Select a game", game_options)

# Dropdown to select a team
if selected_pass_game == "All games":
    team_options = df_pass["Team"].unique()
else:
    filtered_pass_game = df_pass[df_pass["Game"] == selected_pass_game]
    team_options = filtered_pass_game["Team"].unique()

selected_pass_team = st.sidebar.selectbox("Select a team", options=team_options)

# Filter data for the selected team
if selected_pass_game == "All games":
    filtered_pass_team = df_pass[df_pass["Team"] == selected_pass_team]
else:
    filtered_pass_team = filtered_pass_game[filtered_pass_game["Team"] == selected_pass_team]

# Plot the team pass map
st.header(f"Passes for {selected_pass_team}")
plot_pass_maps(selected_pass_team, filtered_pass_team)


from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
def create_team_heatmap(df_pass):
    st.sidebar.header("Select Team for Heatmap")
    st.header("Team Heatmap on Football Pitch")

    game_options = ["All Games"] + df_pass["Game"].unique().tolist()
    selected_game = st.sidebar.selectbox("Select Game:", game_options)

    # Step 1: Team selection
    team_heat = st.sidebar.selectbox("Select a team:", df_pass["Team"].unique())

    # Step 2: Game selection (add a new filter)
    

    # Filter data for the selected team and game
    if selected_game == "All Games":
        team_data = df_pass[df_pass["Team"] == team_heat]
    else:
        team_data = df_pass[(df_pass["Team"] == team_heat) & (df_pass["Game"] == selected_game)]

    # Create the heatmap plot
    fig_heat, ax = plt.subplots(figsize=(6, 4))
    fig_heat.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')
    ax.set_title(f'{team_heat} Team Heatmap - {selected_game}', color='white', fontsize=10, pad=20)

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
        # fill=True,
        alpha=0.7,
        levels=10,
        cmap=custom_cmap,
        n_levels=40,
        shade=True,
        ax=ax
    )

    # Set pitch boundaries
    plt.xlim(0, 120)
    plt.ylim(0, 80)

    # Display the heatmap in Streamlit
    st.pyplot(fig_heat)
    download_plot(fig_heat, f"{team_heat}_heatmap.png", facecolor=pitch.pitch_color)
df_pass = fetch_data("Passes")
create_team_heatmap(df_pass)





st.sidebar.header("Select Stat for Viz top 5")
selected_stat = st.sidebar.selectbox("Select Stat", stats)


df_top5 = df_stats.sort_values(by=selected_stat, ascending=False).head(5)

# Plotting the bar chart
figs, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_top5['Player'], df_top5[selected_stat], color=['blue', 'green', 'red', 'orange', 'purple'])
ax.set_title(f'Top 5 Players by {selected_stat}')
ax.set_xlabel('Player')
ax.set_ylabel(selected_stat)
ax.set_xticklabels(df_top5['Player'], rotation=45)

# add_logo_on_heatmap(ax, "Logo.png", zoom=0.02, x_pos=0.5, y_pos=0.5, alpha=0.15)

st.pyplot(figs)
download_plot(figs, f"Top 5 {selected_stat}.png")


import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt




df_new = fetch_data("shoot")

# import streamlit as st
from mplsoccer import Pitch

def plot_shot_map(df_new):
    st.sidebar.header("Game & Team Shot Analysis")

    # Get unique games from the data

    games = df_new["Game"].unique()

    # Select game
    selected_game = st.sidebar.selectbox("Select Game", games, index=0)

    # Filter data by selected game
    filtered_df = df_new[df_new["Game"] == selected_game]

    # Get unique teams from the filtered data
    teams = filtered_df["team"].unique()

    # Define team colors dynamically based on the teams in the data
    team_colors = {team: f"C{i+1:02d}0{i+1:02d}" for i, team in enumerate(teams)}

    # Team selection
    team1 = st.sidebar.selectbox("Select Team 1", teams, index=0)
    team2 = st.sidebar.selectbox("Select Team 2", teams, index=1)

    # Split df into two parts, one for each team
    team1_df = filtered_df[filtered_df["team"] == team1].copy()
    team2_df = filtered_df[filtered_df["team"] == team2].copy()

    # Split into goals and non-goals for each team
    team1_df_g = team1_df[team1_df["outcome"] == "Goal"].copy()
    team1_df_ng = team1_df[team1_df["outcome"] != "Goal"].copy()

    team2_df_g = team2_df[team2_df["outcome"] == "Goal"].copy()
    team2_df_ng = team2_df[team2_df["outcome"] != "Goal"].copy()

    # Team stats
    team1_tot_shots = team1_df.shape[0]
    team1_tot_goals = team1_df_g.shape[0]
    team1_tot_xg = team1_df["statsbomb_xg"].sum().round(2)

    team2_tot_shots = team2_df.shape[0]
    team2_tot_goals = team2_df_g.shape[0]
    team2_tot_xg = team2_df["statsbomb_xg"].sum().round(2)

    # Plot settings
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white', goal_type='box')

    # Create the figure
    fig, ax = pitch.draw(figsize=(7, 6))

    # Title
    ax.set_title(f"{team1} vs {team2} Shot Map", fontsize=10, pad=5)

    # Plot Team 1 shots (shooting towards the right goal)
    # Goals
    pitch.scatter(team1_df_g["start_location_x"],
                  team1_df_g["start_location_y"],
                  s=team1_df_g["statsbomb_xg"] * 500 + 100,
                  marker="football",
                  c="blue",
                  label=f"{team1} goals",
                  ax=ax)
    # Non-goals
    pitch.scatter(team1_df_ng["start_location_x"],
                  team1_df_ng["start_location_y"],
                  s=team1_df_ng["statsbomb_xg"] * 500 + 100,
                  c="red",
                  alpha=0.5,
                  hatch="//",
                  edgecolor="#101010",
                  marker="s",
                  label=f"{team1} non-goals",
                  ax=ax)

    # Plot Team 2 shots (shooting towards the left goal)
    # Goals
    pitch.scatter(120 - team2_df_g["start_location_x"],  # Mirroring x-coordinates for the other direction
                  team2_df_g["start_location_y"],
                  s=team2_df_g["statsbomb_xg"] * 500 + 100,
                  marker="football",
                  c="cyan",
                  label=f"{team2} goals",
                  ax=ax)
    # Non-goals
    pitch.scatter(120 - team2_df_ng["start_location_x"],  # Mirroring x-coordinates for the other direction
                  team2_df_ng["start_location_y"],
                  s=team2_df_ng["statsbomb_xg"] * 500 + 100,
                  c="orange",
                  alpha=0.5,
                  hatch="//",
                  edgecolor="#101010",
                  marker="s",
                  label=f"{team2} non-goals",
                  ax=ax)

    # Add basic info for each team at the bottom
    basic_info_txt = (
        f"{team1} - Shots: {team1_tot_shots} | Goals: {team1_tot_goals} | xG: {team1_tot_xg}\n"
        f"{team2} - Shots: {team2_tot_shots} | Goals: {team2_tot_goals} | xG: {team2_tot_xg}"
    )
    fig.text(0.5, 0.02, basic_info_txt, fontsize=5, ha="center", color="white")

    # Add legend
    ax.legend(labelspacing=2, loc="upper center", fontsize=5, bbox_to_anchor=(0.5, -0.05), ncol=2)

    # Display the plot
    st.pyplot(fig)
    download_plot(fig, f"{team1}_vs_{team2}_shot_map.png")

# Call the function with your DataFrame
plot_shot_map(df_new)

