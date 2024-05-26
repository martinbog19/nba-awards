import numpy as np
import pandas as pd
from scipy.stats import kendalltau
#import matplotlib.


# Function that evaluates a predictive model on a given test set using 7 metrics

# mae        :  Mean absolute error on all predicted shares
# R2         :  R2 score on all predicted shares -- compares to baseline that predicts mean historical shares for all players
# mae_votes  :  Mean absolute error on predicted shares only for players who actually recieved votes
# tau        :  Kendall's tau ranking score of players who received votes
# acc1       :  Is the winner correctly predicted?
# acc3       :  Accuracy on predicted top 3
# acc5       :  Accuracy on predicted top 5


def model_evaluation(model, features, test_set, y_test, year = None) :

    y_pred = model.predict(np.array(test_set[features]))
    y_pred = y_pred[:,0 ] if y_pred.shape[-1] == 1 else y_pred
    test_set['Pred'] = y_pred

    # Simple MAE and R2
    mae = np.abs(y_pred - y_test).mean()
    r2 = 1 - ((y_test - y_pred) ** 2).sum() / (y_test ** 2).sum()

    # Players with votes
    withVotes = test_set[test_set['Share'] > 0].sort_values('Share', ascending = False).reset_index(drop = True)

    # MAE on players who received votes
    mae_votes = (withVotes['Pred'] - withVotes['Share']).abs().mean()

    # Rankings on players who received votes
    tau, _ = kendalltau(withVotes['Share'], withVotes['Pred'])
    
    # Store actual winner, top-3 and top-5
    winner = withVotes['href'].values[0]
    top3 = withVotes['href'].values[:3].tolist()
    top5 = withVotes['href'].values[:5].tolist()

    # Store predicted winner, top-3 and top-5
    sorted_pred = test_set.sort_values('Pred', ascending = False).reset_index(drop = True)
    pred_winner = sorted_pred['href'].values[0]
    pred_top3 = sorted_pred['href'].values[:3].tolist()
    pred_top5 = sorted_pred['href'].values[:5].tolist()

    # Accuracies on winner, top-3 and top-5
    acc1 = (winner == pred_winner)
    acc3 = len(set(top3).intersection(set(pred_top3))) / 3
    acc5 = len(set(top5).intersection(set(pred_top5))) / 5

    # Assemble results
    winner_name = withVotes['Player'].values[0]
    winner_pred = withVotes['Pred'].values[0]
    winner_pred_rank = (sorted_pred['href'] == winner).argmax() + 1
    pred_winner_name = sorted_pred['Player'].values[0]
    pred_winner_share = sorted_pred['Share'].values[0]
    if pred_winner in withVotes['href'].tolist() :
        pred_winner_rank = (withVotes['href'] == pred_winner).argmax() + 1
    else :
        pred_winner_rank = -1

    res = pd.DataFrame([[year, winner_name, winner_pred, winner_pred_rank, pred_winner_name, pred_winner_share, pred_winner_rank, acc1, r2, mae, mae_votes, tau, acc3, acc5]],
                       columns = ['Year', 'Winner', 'Pred. share', 'Pred. rank', 'Pred. winner', 'Share', 'Rank', 'Correct', 'R2', 'MAE', 'MAE w. votes', 'Tau', 'Top-3', 'Top-5'])

    return res


def backtest(start_year, data, model, features, keras = False) :

    years = np.arange(start_year, 2025)
    dfs = []
    for i, year in enumerate(years) :

        print(f'[{i+1}/{len(years)}] ... {year} ...   ')

        train = data.copy()[data['Year'] < year]
        test  = data.copy()[data['Year'] == year]

        X_train = np.array(train[features])
        y_train = np.array(train['Share'])

        X_test = np.array(test[features])
        y_test = np.array(test['Share'])

        if not keras :
            model.fit(X_train, y_train)
        else :
            model.fit(X_train, y_train, epochs = keras[0], batch_size = keras[1], validation_split = 0.1, verbose = 0)

        res = model_evaluation(model, features, test.copy(), y_test, year)
        dfs.append(res)

    backtests = pd.concat(dfs).set_index('Year')

    return backtests