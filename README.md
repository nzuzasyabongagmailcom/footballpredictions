# Football Predictions App

Streamlit app that predicts goals, scores and corners for the Premier League,
La Liga, Serie A, Bundesliga and Ligue 1.

- **Upcoming**: predictions for the next round, from football-data.co.uk/fixtures.csv
- **Previous**: this season's predictions next to the real results, with hit rates
- **Match detail**: any two teams, full probabilities and a score grid

Models: Dixon-Coles goals model and a shrunk negative binomial corners model
(shrink 0.5, dispersion 1.3), refitted from the latest results every 6 hours.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free on Streamlit Community Cloud
1. Put app.py, model.py and requirements.txt in a public GitHub repository.
2. Go to share.streamlit.io, sign in with GitHub, click "Create app".
3. Pick the repository, branch main, main file app.py, then Deploy.

For analysis and learning. If you bet, only stake what you can afford to lose.
