import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time


def PlayerStats(year) : # Function to scrape player stats for a given year

    def single_team(df) :
        # For an input player, this function returns only a row with total stats and the latest team of the player
        if len(df) > 1 : # If a player played for multiple teams
            row = df[df['Tm'] == 'TOT']
            df = df[df['Tm'] != 'TOT']
            row['Tm'] = df[df['G'] == df['G'].max()]['Tm'].values[0] # Keep team for which player has played the most
            row['LatestTm'] = df.tail(1)['Tm'].values[0] # Store the latest playing team
            return row
        else :
            df['LatestTm'] = df['Tm']
            return df

    # Scrape per game stats -- PTS, TRB, AST, ...
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find('table')
    while table.find(class_ = 'thead') is not None :
        table.find(class_ = 'thead').decompose()
    data_pg = pd.read_html(str(table))[0].drop(columns = ['Rk'])
    data_pg.insert(1, 'href', [str(x).split('.html')[0].split('/')[-1] for x in table.find_all('a', href = True) if 'players' in str(x)])
    data_pg = data_pg.groupby('href').apply(single_team).reset_index(drop = True)
    data_pg['Player'] = data_pg['Player'].str.replace('*', '', regex = False)

    # Scrape advanced stats -- PER, BPM, WS, VORP, ...
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    table = soup.find('table')
    while table.find(class_ = 'thead') is not None :
        table.find(class_ = 'thead').decompose()
    data_adv = pd.read_html(str(table))[0]
    data_adv = data_adv.drop(columns = ['Rk'] + [x for x in data_adv.columns if 'Unnamed' in x])
    data_adv.insert(1, 'href', [str(x).split('.html')[0].split('/')[-1] for x in table.find_all('a', href = True) if 'players' in str(x)])
    data_adv = data_adv.groupby('href').apply(single_team).reset_index(drop = True)
    data_adv['Player'] = data_adv['Player'].str.replace('*', '', regex = False)

    # Merge per game and advanced stats together
    data = data_pg.merge(data_adv, on = ['Player', 'href'], suffixes = ('', '_y'))
    data = data.drop(columns = [col for col in data.columns if '_y' in col]).reset_index(drop = True) # Delete duplicated columns
    data.insert(2, 'Year', len(data) * [year])

    # Check that merge is 1-to-1
    if len(data) != len(data_pg) or len(data) != len(data_adv):
        Warning('Merge between per game and advanced stats is not 1:1 !!')

    return data


def TeamStats(year) : # Function to scrape team stats for a given year

    # Scrape team standings
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')

    while soup.find('tr', class_ = 'thead') is not None:
        soup.find('tr', class_ = 'thead').decompose()

    # Eastern conference
    table_E = soup.find('table', id = 'divs_standings_E')
    teams_E = pd.read_html(str(table_E))[0]
    teams_E = teams_E.rename(columns = {'Eastern Conference': 'Team'})
    teams_E['Tm'] = [x['href'].split('/')[2] for x in table_E.find_all('a', href = True)]
    teams_E['Seed'] = [len(teams_E[teams_E['W/L%'] > wl]) + 1 for wl in teams_E['W/L%']]

    # Western conference
    table_W = soup.find('table', id = 'divs_standings_W')
    teams_W = pd.read_html(str(table_W))[0]
    teams_W = teams_W.rename(columns = {'Western Conference': 'Team'})
    teams_W['Tm'] = [x['href'].split('/')[2] for x in table_W.find_all('a', href = True)]
    teams_W['Seed'] = [len(teams_W[teams_W['W/L%'] > wl]) + 1 for wl in teams_W['W/L%']]

    # Assemble league-wise table
    teams = pd.concat([teams_E, teams_W]).sort_values('W/L%', ascending = False).reset_index(drop = True)
    teams = teams.drop(columns = ['GB', 'PS/G', 'PA/G'])
    teams['Team'] = teams['Team'].str.replace('*', '', regex = False)
    teams['Year'] = len(teams) * [year]

    # Scrape team ratings
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_ratings.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    soup.find('tr', class_='over_header').decompose()    
    table = soup.find('table')
    ratings = pd.read_html(str(table))[0]
    ratings['Tm'] = [x['href'].split('/')[2] for x in table.find_all('a', href = True)]
    ratings['Year'] = len(teams) * [year] # Keep track of year
    ratings = ratings[['Team', 'Tm', 'Year', 'MOV', 'ORtg', 'DRtg', 'NRtg']]

    # Merge standings and ratings data
    data = teams.merge(ratings.drop(columns = ['Team']), on = ['Tm', 'Year'])

    return data


def AwardShares(year, award) : # Function to scrape award shares for a given year

    url = f'https://www.basketball-reference.com/awards/awards_{year}.html'
    page = requests.get(url)
    if f'"div_{award}"' in page.text :
        table = page.text.split(f'<div class="table_container" id="div_{award}">')[1].split('</table>')[0] + '</table>'
    else :
        table = page.text.split(f'<div class="table_container" id="div_nba_{award}">')[1].split('</table>')[0] + '</table>'
    soup = BeautifulSoup(table, 'lxml')
    soup.find('tr', class_ = 'over_header').decompose()
    df = pd.read_html(str(soup))[0]
    df['href'] = [x['href'].split('.html')[0].split('/')[-1] for x in soup.find_all('a', href = True) if 'players' in x['href']]
    df['Year'] = len(df) * [year]
    df = df[['Player', 'href', 'Year', 'Share']]

    return df


def Rookies(year) : # Function to scrape award shares for a given year

    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_rookies.html'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    for header in ['thead', 'over_header'] :
        while soup.find('tr', class_ = header) is not None :
            soup.find('tr', class_ = header).decompose()
    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    df['href'] = [x['href'].split('/')[-1].split('.')[0] for x in table.find_all('a', href = True) if 'players' in x['href']]
    df['Year'] = year
    df = df[['Player', 'href', 'Year']]

    return df



# Create a dataframe with all player stats from 1970-71 to 2023-24
# years = np.arange(1971, 2025)
# dfs = []
# for year in years :

#     print(f'Fetching player stats for season {year-1}-{year} ...', end = '\r')
#     stats   = PlayerStats(year)
#     ratings = TeamStats(year)
#     df = stats.merge(ratings, on = ['Tm', 'Year'], how = 'left')
#     dfs.append(df)
#     time.sleep(5)

# data = pd.concat(dfs)
# data = data[['Player', 'href', 'Year', 'Team', 'Tm', 'LatestTm', 'Pos', 'Age', 'G', 'GS', 'MP', 'FG',
#        'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT',
#        'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
#        'PTS', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
#        'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48',
#        'OBPM', 'DBPM', 'BPM', 'VORP', 'W', 'L', 'W/L%', 'SRS', 'Seed',
#        'MOV', 'ORtg', 'DRtg', 'NRtg']]
# data.to_csv('data/PlayersStats_1971-2024.csv', index = None)


# Create a dataframe with all MVP voting from 1970-71 to 2023-24
# years = np.arange(1971, 2025)
# dfs = []
# for year in years :

#     print(f'Fetching MVP shares for season {year-1}-{year} ...', end = '\r')
#     dfs.append(AwardShares(year, 'mvp'))
#     time.sleep(5)

# data = pd.concat(dfs).reset_index(drop = True)
# data.to_csv('data/SharesMVP_1971-2024.csv', index = None)

# Create a dataframe with all DPOY voting from 1982-83 to 2023-24
# years = np.arange(1983, 2025)
# dfs = []
# for year in years :

#     print(f'Fetching DPOY shares for season {year-1}-{year} ...', end = '\r')
#     dfs.append(AwardShares(year, 'dpoy'))
#     time.sleep(5)

# data = pd.concat(dfs).reset_index(drop = True)
# data.to_csv('data/SharesDPOY_1983-2024.csv', index = None)

# Create a dataframe with all ROY voting from 1970-71 to 2023-24
# years = np.arange(1971, 2025)
# dfs = []
# for year in years :

#     print(f'Fetching ROY shares for season {year-1}-{year} ...', end = '\r')
#     dfs.append(AwardShares(year, 'roy'))
#     time.sleep(5)

# data = pd.concat(dfs).reset_index(drop = True)
# data.to_csv('data/SharesROY_1971-2024.csv', index = None)

# Create a dataframe with all SMOY voting from 1983-84 to 2023-24
# years = np.arange(1984, 2025)
# dfs = []
# for year in years :

#     print(f'Fetching SMOY shares for season {year-1}-{year} ...', end = '\r')
#     dfs.append(AwardShares(year, 'smoy'))
#     time.sleep(5)

# data = pd.concat(dfs).reset_index(drop = True)
# data.to_csv('data/SharesSMOY_1984-2024.csv', index = None)

# Create a dataframe with all MIP voting from 1985-86 to 2023-24
# years = np.arange(1986, 2025)
# dfs = []
# for year in years :

#     print(f'Fetching MIP shares for season {year-1}-{year} ...', end = '\r')
#     dfs.append(AwardShares(year, 'mip'))
#     time.sleep(5)

# data = pd.concat(dfs).reset_index(drop = True)
# data.to_csv('data/SharesMIP_1986-2024.csv', index = None)

# Create a dataframe with all rookies from 1970-71 to 2023-2024
years = np.arange(1971, 2025)
dfs = []
for year in years :

    print(f'Fetching list of rookies for season {year-1}-{year} ...', end = '\r')
    dfs.append(Rookies(year))
    time.sleep(5)

data = pd.concat(dfs).reset_index(drop = True)
data.to_csv('data/Rookies_1971-2024.csv', index = None)