import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import mplsoccer
from mplsoccer.pitch import Pitch

# Google Sheets document ID
# https://docs.google.com/spreadsheets/d/11GOW9_pzJmAAAlvWKFYdh7YCc0lF4U7w/edit?gid=306235239#gid=306235239

# st. set_page_config(layout="wide")
st.set_page_config( page_title="Lagos Liga Analysis With MIAS")

# document_id = '1oCS-ubjn2FtmkHevCToSCfcgL6WpjgXA3qoGnsu8IWk'

# def fetch_data(sheet_name):
#     url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    # return pd.read_csv(url)
# document_id = '1XDRCoTNodcU28nk4HYxOzjBJzt_e5gEo'

# # @st.cache_data
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

df2 = df2[['match_id_new','team_name','Goals','Goal Attempts','Shots On Target','Shots Off Target', 'Fouls', 'Corners', 'Yellow Card', 'Red Card','Goalkeeper Saves']]


match_id = df2['match_id_new'].unique()
st.sidebar.header("Match Statistics")
item = st.sidebar.selectbox("Select match ID", match_id)

match_id_filtered = df2[df2['match_id_new'] == item]


fig1 = create_plot(match_id_filtered, f'{match_id_filtered["team_name"].iloc[0]} vs {match_id_filtered["team_name"].iloc[1]}')
st.title("Lagos Liga Analysis With MIAS")
st.subheader("Match Day Team Stats")
st.plotly_chart(fig1)



def create_radar_chart(df, player1, player2):
    metrics = df.columns[1:].tolist()
    
    player1_data = df[df['Player'] == player1].iloc[0, 1:].tolist()
    player2_data = df[df['Player'] == player2].iloc[0, 1:].tolist()

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

    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns

# def plot_pass_map_for_player(ax, player_name, df_pass):
#     # Filter the pass DataFrame for the selected player
#     player_pass_df = df_pass[df_pass['player'] == player_name]

#     # Create the pitch
#     pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
#     pitch.draw(ax=ax)
#     ax.set_title(f'Pass Map for {player_name}', color='white', fontsize=10, pad=20)
#     ax.invert_yaxis()

#     # Plot each pass
#     for _, row in player_pass_df.iterrows():
#         if row['outcome'] == 'successful':
#             ax.plot((row['x'], row['endX']), (row['y'], row['endY']), color='green')
#             ax.scatter(row['x'], row['y'], color='green')
#         elif row['outcome'] == 'unsuccessful':
#             ax.plot((row['x'], row['endX']), (row['y'], row['endY']), color='red')
#             ax.scatter(row['x'], row['y'], color='red')

# def plot_pass_maps(player1, player2, df_pass):
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two plots side by side

#     fig.set_facecolor('#22312b')
#     axs[0].patch.set_facecolor('#22312b')
#     axs[1].patch.set_facecolor('#22312b')

#     plot_pass_map_for_player(axs[0], player1, df_pass)
#     plot_pass_map_for_player(axs[1], player2, df_pass)

#     st.pyplot(fig)
from matplotlib.lines import Line2D 
from matplotlib.lines import Line2D  # Import Line2D for custom legend elements
import io
from matplotlib import pyplot as plt
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

# def plot_pass_maps(player1, df_pass):
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two plots side by side

#     fig.set_facecolor('#22312b')
#     axs[0].patch.set_facecolor('#22312b')
#     axs[1].patch.set_facecolor('#22312b')

#     plot_pass_map_for_player(axs[0], player1, df_pass)
#     # plot_pass_map_for_player(axs[1], player2, df_pass)

#     st.pyplot(fig)
#     buffer = io.BytesIO()
#     fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='#22312b')
#     buffer.seek(0)

#     # Add a download button in Streamlit
#     st.download_button(
#         label="Download Pass Maps",
#         data=buffer,
#         file_name=f"{player1}_vs_{player2}_pass_maps.png",
#         mime="image/png"
#     )


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
    st.pyplot(fig)

    # Save the figure to a buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='#22312b')
    buffer.seek(0)

    # Add a download button in Streamlit
    st.download_button(
        label="Download Pass Map",
        data=buffer,
        file_name=f"{player1}_pass_map.png",
        mime="image/png"
    )








df = fetch_data('PLAYER_STATS')
df = df.iloc[:, 5:60]

df_stats = df.iloc[:,3:55]
df_stats = df_stats.groupby('Player').sum(numeric_only=True).reset_index()


stats = ['Goals', 'Assists', 'Chances Created','Shots on target','Shots off target', 'Crosses', 'Fouls committed', 'Fouls drawn', 'Corner', 'Saves', 'Interception', 'Pass complete', 'Key Passes', 'Goal Involvement']


df_pass = fetch_data("Passes")

# st.sidebar.title('Player Comparison Radar and Passes Chart')
st.sidebar.header("Radar Chart Comparison")

# Filter by position


# Filter the radar DataFrame by the selected position



team_options = df['Team'].unique()
team1 = st.sidebar.selectbox("Select Team 1", team_options, index=4)
team2 = st.sidebar.selectbox("Select Team 2", team_options, index=5)

# Filter the DataFrame based on selected teams

position_options_team = df['POS'].unique()
selected_position = st.sidebar.selectbox("Select Position", position_options_team)

# if selected_position == 'GK':
#     columns_to_use = ['Player', 'Save Percentage', 'Clean Sheets','Goal Conceded', 'Saves', 'Crosses Stopped', 'Penalty saves', 'Cross Clamed', 'Sweeper actions' ]                 
# elif selected_position == 'FW':
#     columns_to_use = ['Player', 'Goals', 'Shots on target', 'Conversion rate', 'Assists','Goal involvement', 'Penalties won', 'Attacking contribution', 'Minutes per goal']
# elif selected_position == 'MF':
#     columns_to_use = ['Player', 'Goals', 'Assists', 'Tackles', 'Interceptions', 'Chances created', 'Defensive errors','Fouls committed','Blocks','Goal involvement']      
# elif selected_position == 'DF':
#     columns_to_use = ['Player', 'Goals', 'Tackles', 'Fouls Won', 'Blocks', 'Minutes per card','Goal involvement', 'Defensive contribution', 'Defensive errors']        


if selected_position == 'GK':
    columns_to_use = ['Player',  'Clean sheet', 'Saves', 'Crosses Claimed', 'Clearances']                 
elif selected_position == 'FW':
    columns_to_use = ['Player', 'Goals', 'Shots on target',  'Assists', 'Chances Created', 'Pass complete', 'Key Passes']#'Attacking Contribution'
elif selected_position == 'MF':
    columns_to_use = ['Player', 'Goals', 'Assists', 'Tackles', 'Interceptions', 'Chances created', 'Key Passes','Fouls committed','Blocks','Goal involvement']      
elif selected_position == 'DF':
    columns_to_use = ['Player', 'Goals', 'Tackles', 'Interceptions', 'Blocks', 'Clearances','Fouls committed', 'Defensive Contribution', 'Error']        

df_team1_filtered = df[df['Team'] == team1]
df_team2_filtered = df[df['Team'] == team2]


filtered_df_radar = df[df['POS'] == selected_position]
filtered_df_radar = df[columns_to_use]

# Get all unique players from the selected teams
players_team1 = df_team1_filtered['Player'].unique().tolist()
players_team2 = df_team2_filtered['Player'].unique().tolist()

player1 = st.sidebar.selectbox(f"Select Player from {team1}", players_team1, index=1)
player2 = st.sidebar.selectbox(f"Select Player from {team2}", players_team2, index=1)



# Ensure two different players are selected
if player1 != player2:
    # st.sidebar.subheader('Player Stats')
    player1_value = filtered_df_radar[filtered_df_radar['Player'] == player1]
    player2_value = filtered_df_radar[filtered_df_radar['Player'] == player2]
    
    # Merge player data for display and plotting
    merged_df = pd.concat([player1_value, player2_value], ignore_index=True)
    st.write(merged_df)
    # st.dataframe(merged_df, use_container_width=True)  


    st.header('Radar Chart Comparison')
    fig = create_radar_chart(filtered_df_radar, player1, player2)
    st.plotly_chart(fig)
else:
    st.sidebar.warning("Please select two different players.")


st.sidebar.header("Pass Charts")
selected_pass_game = st.sidebar.selectbox("Select a game", df_pass["Game"].unique())

# Filter the DataFrame by selected game
filtered_pass_game = df_pass[df_pass["Game"] == selected_pass_game]

# Dropdown to select a team
selected_pass_team = st.sidebar.selectbox("Select a team", options=filtered_pass_game["Team"].unique())

# Filter the DataFrame by selected team
filtered_pass_team = filtered_pass_game[filtered_pass_game["Team"] == selected_pass_team]

# Dropdowns to select Player 1 and Player 2
players = filtered_pass_team["player"].unique()
player1_pass = st.sidebar.selectbox("Select Player 1", options=players, key="player1", index = 1)
# player2_pass = st.selectbox("Select Player 2", options=players, key="player2", index = 1)
st.header("Player Passes on Football Pitch")
plot_pass_maps(player1_pass, df_pass)




st.sidebar.header("Select Team and player for Heatmap")
st.header(" Player Heatmap on Football Pitch")
# Step 1: Team selection
team_heat = st.sidebar.selectbox("Select a team:", df_pass["Team"].unique())

# Filter players based on the selected team
team_players_heat = df_pass[df_pass["Team"] == team_heat]["player"].unique()

# Step 2: Player selection
player_heat = st.sidebar.selectbox("Select a player from the team:", team_players_heat)

# Filter data for the selected player
player_data = df_pass[(df_pass["Team"] == team_heat) & (df_pass["player"] == player_heat)]

# Create the heatmap plot
fig_heat, ax = plt.subplots(figsize=(6, 4))
fig_heat.set_facecolor('#22312b')
ax.patch.set_facecolor('#22312b')

# Create the pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
pitch.draw(ax=ax)
plt.gca().invert_yaxis()


sns.kdeplot(
    x=player_data['x'],
    y=player_data['y'],
    fill=True,  # Use fill instead of shade
    alpha=0.5,
    levels=10,
    cmap='magma',
    ax=ax  # Ensure you pass the Axes to Seaborn
)

# Set pitch boundaries
plt.xlim(0, 120)
plt.ylim(0, 80)

# Display the heatmap in Streamlit
st.pyplot(fig_heat)
import io
from matplotlib import pyplot as plt
buffer = io.BytesIO()
fig_heat.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor=pitch.pitch_color)
buffer.seek(0)

# Add a download button in Streamlit
st.download_button(
    label="Download",
    data=buffer,
    file_name=f"{player_heat}_heatmap.png",
    mime="image/png"
)


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


st.pyplot(figs)

import io
from matplotlib import pyplot as plt

# ... (your existing code to create the figure and plot data)

# Save the plot to a BytesIO buffer
buffer = io.BytesIO()
figs.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor=pitch.pitch_color)
buffer.seek(0)

# Add a download button in Streamlit
st.download_button(
    label="Download Stats",
    data=buffer,
    file_name=f"Top 5 {selected_stat}.png",
    mime="image/png"
)


import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt

# Load data

df_new = fetch_data("shoot")

# Streamlit interface
st.sidebar.header("Team Shot Analysis")

# Get unique teams from the data
teams = df_new["team"].unique()

# Define team colors dynamically based on the teams in the data
team_colors = {team: f"C{i+1:02d}0{i+1:02d}" for i, team in enumerate(teams)}

# Team selection
team1 = st.sidebar.selectbox("Select Team 1", teams, index=0)
team2 = st.sidebar.selectbox("Select Team 2", teams, index = 1)

# Split df into two parts, one for each team
team1_df = df_new[df_new["team"] == team1].copy()
team2_df = df_new[df_new["team"] == team2].copy()

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

# Plotting
# pitch = VerticalPitch(half=True, pitch_color="green", line_color="white")

# fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# # Plot for Team 1
# pitch.draw(ax=ax[0])
# ax[0].set_title(f"{team1} Shots vs {team2}")

# # Team 1 goals
# pitch.scatter(team1_df_g["start_location_x"],
#               team1_df_g["start_location_y"],
#               s=team1_df_g["statsbomb_xg"]*500+100,
#               marker="football",
#               c="blue",
#               ax=ax[0],
#               label=f"{team1} goals")
# # Team 1 non-goals
# pitch.scatter(team1_df_ng["start_location_x"],
#               team1_df_ng["start_location_y"],
#               s=team1_df_ng["statsbomb_xg"]*500+100,
#               c="red",
#               alpha=0.5,
#               hatch="//",
#               edgecolor="#101010",
#               marker="s",
#               ax=ax[0],
#               label=f"{team1} non-goals")

# # Plot for Team 2
# pitch.draw(ax=ax[1])
# ax[1].set_title(f"{team2} Shots vs {team1}")

# # Team 2 goals
# pitch.scatter(team2_df_g["start_location_x"],
#               team2_df_g["start_location_y"],
#               s=team2_df_g["statsbomb_xg"]*500+100,
#               marker="football",
#               c="blue",
#               ax=ax[1],
#               label=f"{team2} goals")
# # Team 2 non-goals
# pitch.scatter(team2_df_ng["start_location_x"],
#               team2_df_ng["start_location_y"],
#               s=team2_df_ng["statsbomb_xg"]*500+100,
#               c='red',
#               alpha=0.5,
#               hatch="//",
#               edgecolor="#101010",
#               marker="s",
#               ax=ax[1],
#               label=f"{team2} non-goals")

# # Team 1 stats
# basic_info_txt1 = f"Shots: {team1_tot_shots} | Goals: {team1_tot_goals} | xG: {team1_tot_xg}"
# ax[0].text(0.5, -0.1, basic_info_txt1, size=15, ha="center", transform=ax[0].transAxes)

# # Team 2 stats
# basic_info_txt2 = f"Shots: {team2_tot_shots} | Goals: {team2_tot_goals} | xG: {team2_tot_xg}"
# ax[1].text(0.5, -0.1, basic_info_txt2, size=15, ha="center", transform=ax[1].transAxes)

# # Legends
# ax[0].legend(labelspacing=1.5, loc="lower center")
# ax[1].legend(labelspacing=1.5, loc="lower center")

# # Display the plots

# st.pyplot(fig)

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
import io
from matplotlib import pyplot as plt

# ... (your existing code to create the figure and plot data)

# Save the plot to a BytesIO buffer
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor=pitch.pitch_color)
buffer.seek(0)

# Add a download button in Streamlit
st.download_button(
    label="Download",
    data=buffer,
    file_name=f"{team1}_vs_{team2}_shot_map.png",
    mime="image/png"
)



data = fetch_data('matches')


def generate_result_string(last_n_team_games):
    return ''.join(last_n_team_games['result'].values)
# Step 10: Data Visualization functions
def filter_team_games(data, team):
    return data[(data['home_team_name'] == team) | (data['away_team_name'] == team)]

def get_last_n_games(team_games, n=3):
    # Filter out games that haven't been played yet (missing goals)
    completed_games = team_games.dropna(subset=['home_team_score', 'away_team_score'])
    return completed_games.tail(n)

def determine_result(row, team):
    if team == row['home_team_name']:
        if row['home_team_score'] > row['away_team_score']:
            return 'W'
        elif row['home_team_score'] < row['away_team_score']:
            return 'L'
        else:
            return 'D'
    elif team == row['away_team_name']:
        if row['away_team_score'] > row['home_team_score']:
            return 'W'
        elif row['away_team_score'] < row['home_team_score']:
            return 'L'
        else:
            return 'D'
    else:
        return None

def display_last_n_games(last_n_team_games):
    columns_to_display = ['home_team_name', 'away_team_name', 'home_team_score', 'away_team_score', 'result']
    return last_n_team_games[columns_to_display]



# Get unique team names
unique_home_teams = data['home_team_name'].unique()
unique_away_teams = data['away_team_name'].unique()

# Remove home teams already present in the away teams list
unique_away_teams = [team for team in unique_away_teams if team not in unique_home_teams]

# Combine unique home and away team names
all_unique_teams = list(unique_home_teams) + list(unique_away_teams)

st.sidebar.header("Team Runs")

# Use the multiselect widget with the combined unique team names
selected_teams = st.sidebar.multiselect("Select Teams", all_unique_teams)

# Ensure exactly two teams are selected

if len(selected_teams) != 2:
    st.sidebar.warning("Please select exactly two teams.")
else:
    # Unpack selected teams
    team1, team2 = selected_teams

    # Step 6: Filter the data based on the selected teams
    team1_games = filter_team_games(data, team1)
    team2_games = filter_team_games(data, team2)

    # Get the last 5 games for each team
    last_5_team1_games = get_last_n_games(team1_games)
    last_5_team2_games = get_last_n_games(team2_games)

    # Determine results for each team
    last_5_team1_games['result'] = last_5_team1_games.apply(determine_result, axis=1, team=team1)
    last_5_team2_games['result'] = last_5_team2_games.apply(determine_result, axis=1, team=team2)

    # Generate result strings
    team1_results_str = generate_result_string(last_5_team1_games)
    team2_results_str = generate_result_string(last_5_team2_games)

    # Display results and games
    st.subheader(f"Last 5 games involving {team1}: {team1_results_str}")
    st.write(display_last_n_games(last_5_team1_games))

    st.subheader(f"Last 5 games involving {team2}: {team2_results_str}")
    st.write(display_last_n_games(last_5_team2_games))



