import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.layers import SimpleRNN
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None  # default='warn'
import yfinance as yf
import plotly.express as px

app = dash.Dash()
server = app.server

def create_feature_ROC(df, n_period = 10):
    df['ROC'] = np.nan
    for i in range (n_period, len(df)):
        df['ROC'][i] = (df['Close'][i] - df['Close'][i-n_period])/df['Close'][i-n_period]*100
    return df

def LSTM_prediction(df, split_ratio = 0.7, ROC = False, n_period = 10, epochs = 40, batch_size = 32, units = 50):
    target = ['Close']
    features = []
    if ROC:
        features.append('ROC')
        df = create_feature_ROC(df, n_period)
    
    df = df[target + features]
    
    # remove na
    df = df.dropna()
    
    # split data
    split = int(df.shape[0]*split_ratio)
    train = df.iloc[:split, :].copy()
    test = df.iloc[split:, :].copy()
    
    # scale data
    target_scaler = MinMaxScaler().fit(train[target])
    train[target] = target_scaler.transform(train[target])
    test[target] = target_scaler.transform(test[target])
    
    if features:
        features_scaler = MinMaxScaler().fit(train[features])
        train[features] = features_scaler.transform(train[features])
        test[features] = features_scaler.transform(test[features])
    
    # input sequence length
    sequence_length = 30
    
    x_train, y_train = [], []
    
    for i in range (sequence_length, train.shape[0]):
        x_train.append(train[features + target].iloc[i - sequence_length: i])
        y_train.append(train[target].iloc[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    x_test, y_test = [], []
    
    for i in range(sequence_length, test.shape[0]):
        x_test.append(test[features + target].iloc[i - sequence_length: i])
        y_test.append(test[target].iloc[i])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # build and train model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    model.evaluate(x_test, y_test)
    
    # generate the test set predictions
    y_pred = model.predict(x_test)
    y_pred = target_scaler.inverse_transform(y_pred)
    
    # add predicted values
    df['Predicted Close'] = np.nan
    df['Predicted Close'].iloc[- y_pred.shape[0]:] = y_pred.flatten()
    
    return df

def RNN_prediction(df, split_ratio = 0.7, ROC = False, n_period = 10, epochs = 40, batch_size = 32, units = 50):
    target = ['Close']
    features = []
    if ROC:
        features.append('ROC')
        df = create_feature_ROC(df, n_period)
    
    df = df[target + features]
    
    # remove na
    df = df.dropna()
    
    # split data
    split = int(df.shape[0]*split_ratio)
    train = df.iloc[:split, :].copy()
    test = df.iloc[split:, :].copy()
    
    # scale data
    target_scaler = MinMaxScaler().fit(train[target])
    train[target] = target_scaler.transform(train[target])
    test[target] = target_scaler.transform(test[target])
    
    if features:
        features_scaler = MinMaxScaler().fit(train[features])
        train[features] = features_scaler.transform(train[features])
        test[features] = features_scaler.transform(test[features])
    
    # input sequence length
    sequence_length = 30
    
    x_train, y_train = [], []
    
    for i in range (sequence_length, train.shape[0]):
        x_train.append(train[features + target].iloc[i - sequence_length: i])
        y_train.append(train[target].iloc[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    x_test, y_test = [], []
    
    for i in range(sequence_length, test.shape[0]):
        x_test.append(test[features + target].iloc[i - sequence_length: i])
        y_test.append(test[target].iloc[i])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # build and train model
    model = Sequential()
    model.add(SimpleRNN(units=units, return_sequences=False, input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    model.evaluate(x_test, y_test)
    
    # generate the test set predictions
    y_pred = model.predict(x_test)
    y_pred = target_scaler.inverse_transform(y_pred)
    
    # add predicted values
    df['Predicted Close'] = np.nan
    df['Predicted Close'].iloc[- y_pred.shape[0]:] = y_pred.flatten()
    
    return df

def XGBoost_prediction(df, split_ratio = 0.7, ROC = False, n_period = 10, learning_rate = 0.2):
    target = ['Close']
    features = ['Adj Close']
    if ROC:
        features.append('ROC')
        df = create_feature_ROC(df, n_period)
    
    df = df[target + features]
    
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()
    
    features = features + ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30',]
    
    # remove na
    df = df.dropna()
    
    # split data
    split = int(df.shape[0]*split_ratio)
    train = df.iloc[:split, :].copy()
    test = df.iloc[split:, :].copy()
    
    x_train = train[features].copy()
    y_train = train[target].copy()
    
    x_test = test[features].copy()
    y_test = test[target].copy()
    
    # build and train model
    eval_set = [(x_train, y_train), (x_test, y_test)]
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=learning_rate, max_depth=10, min_child_weight=1, objective='reg:squarederror')
    model.fit(x_train, y_train, eval_set=eval_set, verbose=100)

    # generate the test set predictions
    y_pred = model.predict(x_test)
    
    # add predicted values
    df['Predicted Close'] = np.nan
    df['Predicted Close'].iloc[- y_pred.shape[0]:] = y_pred.flatten()
    
    return df

app.layout = html.Div([
   
    html.H1("Cryptocurrencies Price Analysis Dashboard", style={"textAlign": "center"}),
   
        html.Div(children=[
            dcc.Input(id="ticker", type="text", placeholder="Type in Token name", value="BTC-USD", style={"width": "80%", 'padding': '20px', 'font-size': '20px', 'margin': 'auto'}),
            html.Div(children=[
                dcc.Checklist(id='features', options=[
                    {'label': 'Rate of change', 'value': 'ROC'},
                ],
                ),
                dcc.Input(id="n-period", type="number", value="10", style={}),
            ], style={'margin': 'auto', 'width':'80%', 'display': 'flex', 'flex-wrap': 'nowrap', 'justify-content': 'space-between', 'padding': '20px'}),
        ], style={'margin': 'auto', 'width':'50%', 'display': 'flex', 'flex-wrap': 'nowrap', 'justify-content': 'space-around', 'padding': '20px', 'flex-direction': 'column'}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='XGBoost',children=[
            html.Div([
                html.H2("Close Price predicted using XGBoost",style={"textAlign": "center"}),
                dcc.Graph(id="xgb-graph"),               
            ])                
        ]),

        dcc.Tab(label='RNN',children=[
            html.Div([
                html.H2("Close Price predicted using RNN",style={"textAlign": "center"}),
                dcc.Graph(id="rnn-graph"),             
            ])                
        ]),

        dcc.Tab(label='LSTM',children=[
            html.Div([
                html.H2("Close Price predicted using LSTM",style={"textAlign": "center"}),
                dcc.Graph(id="lstm-graph"),            
            ])                
        ]),
    ])
])

@app.callback(
    Output("xgb-graph", "figure"), 
    Output("rnn-graph", "figure"), 
    Output("lstm-graph", "figure"), 
    Input("ticker", "value"),
    Input("features", 'value'),
    Input("n-period", 'value'))
def update_line_charts(ticker_value, features, n_period):
    dataframe = yf.download(ticker_value)
    
    useROC = False
    
    if features: 
        if 'ROC' in features:
            useROC = True

    XGB_predicted_dataframe = XGBoost_prediction(dataframe, split_ratio = 0.9, ROC = useROC, n_period = int(n_period), learning_rate = 0.2)
    RNN_predicted_dataframe = RNN_prediction(dataframe, split_ratio = 0.9, ROC = useROC, n_period = int(n_period), epochs = 10, batch_size = 32, units = 50)
    LSTM_predicted_dataframe = LSTM_prediction(dataframe, split_ratio = 0.9, ROC = useROC, n_period = int(n_period), epochs = 10, batch_size = 32, units = 50)

    df = px.data.gapminder()
    
    xgb_fig = px.line(XGB_predicted_dataframe, 
        x=XGB_predicted_dataframe.index, y=[XGB_predicted_dataframe['Close'], XGB_predicted_dataframe['Predicted Close']])
    rnn_fig = px.line(RNN_predicted_dataframe, 
        x=RNN_predicted_dataframe.index, y=[RNN_predicted_dataframe['Close'], RNN_predicted_dataframe['Predicted Close']])
    lstm_fig = px.line(LSTM_predicted_dataframe, 
        x=LSTM_predicted_dataframe.index, y=[LSTM_predicted_dataframe['Close'], LSTM_predicted_dataframe['Predicted Close']])

    return xgb_fig, rnn_fig, lstm_fig

if __name__=='__main__':
    app.run_server(debug=True)