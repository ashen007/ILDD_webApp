import dash

app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                assets_folder='assets/')

server = app.server