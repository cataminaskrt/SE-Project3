#Libraries necessary for the implementation of the dashboard
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from dash.exceptions import PreventUpdate

#CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

white_text_style = {'color': 'white'}

#Loading the Raw Data
df_total = pd.read_csv("df_total.csv", index_col=0, parse_dates=True)
columns = df_total.columns.tolist()
start_date = df_total.index.min()
end_date = df_total.index.max()

# Cut-off date for the 2019 set
testing_cutoff_date = '2024-01-01'

#Splitting the Raw Data between training and testing data
training_data = df_total.loc[df_total.index < testing_cutoff_date] #dataset with values from 2022 and 2023
testing_data = df_total.loc[df_total.index >= testing_cutoff_date] #dataset with values from 2024

#Cleaning the training set
training_data = training_data.dropna() 

#Creating and cleaning a separate dataset for meteorological data in 2022,2023
df_meteo = training_data.copy()
df_meteo = df_meteo.drop("Solar Power Aggregated [MW]", axis=1)

#Creating a more complete dataset for feature selection 
df_meteo_complete = df_meteo.copy()
def season(month):
    if month in [12, 1, 2]:  
        return 0
    elif month in [3, 4, 5]:  
        return 1
    elif month in [6, 7, 8]:  
        return 1
    else:  
        return 1
df_meteo_complete['Month']=df_meteo_complete.index.month
df_meteo_complete['Hot Seasons'] = df_meteo_complete['Month'].apply(season)
df_meteo_complete['Solar Power-1h']=training_data['Solar Power Aggregated [MW]'].shift(1) # Previous hour production
df_meteo_complete['Solar Power-2h']=df_meteo_complete['Solar Power-1h'].shift(1) # Second previous hour production
df_meteo_complete['Hour'] = df_meteo_complete.index.hour
df_meteo_complete = df_meteo_complete.dropna()

#Creating a more complete training set dataset for after feature selection
training_data_complete = training_data.copy()
training_data_complete['Month']=training_data_complete.index.month
training_data_complete['Hot Seasons'] = training_data_complete['Month'].apply(season)
training_data_complete['Solar Power-1h']=training_data['Solar Power Aggregated [MW]'].shift(1) # Previous hour production
training_data_complete['Solar Power-2h']=training_data_complete['Solar Power-1h'].shift(1) # Second previous hour production
training_data_complete['Hour'] = training_data_complete.index.hour
training_data_complete = training_data_complete.dropna()

#Loading the real values for 2024
df_2024 = pd.read_csv('df_2024.csv')
y=df_2024['Solar Power Aggregated [MW]'].values

#Creating a separate dataset just with meteorological data for 2024
df_meteo_2024 = df_2024.copy()
df_meteo_2024 = df_meteo_2024.drop("Solar Power Aggregated [MW]", axis=1)
df_meteo_2024 = df_meteo_2024.dropna()
df_meteo_2024['Date_Hour'] = pd.to_datetime(df_meteo_2024['Date_Hour'])
df_meteo_2024.set_index('Date_Hour', inplace=True)


#Cleaning and adding a new dataset to visualize raw data of 2024
df_2024_visualize = df_2024.drop(columns=['Solar Power-1h', 'Solar Power-2h', 'Hot Seasons', 'Hour', 'Month'])
df_2024['Date_Hour'] = pd.to_datetime(df_2024['Date_Hour'])
df_2024.set_index('Date_Hour', inplace=True)
df_2024_visualize = df_2024.drop(columns=['Solar Power-1h', 'Solar Power-2h', 'Hot Seasons', 'Hour', 'Month'])

#Initializing the variables
X = None
Y = None

X_train = None
X_test = None
y_train = None
y_test = None

X_2019 = None
y_pred2024 = None
x_2024 = None
y_2024 = None
X_2024 = None
Y_2024 = None

x_day_ahead = None
y_day_ahead = None
#By default, plot the real values of Solar Power for 2024
fig2 = px.line(df_2024, x=df_2024.index, y='Solar Power Aggregated [MW]')

#Auxiliary functions
def generate_table(dataframe, max_rows=10):
    # Apply some CSS styles to the table
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }
    
    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }
    
    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    return html.Table(
        # Apply the table style
        style=table_style,
        children=[
            # Add the table header
            html.Thead(
                html.Tr([
                    *[html.Th(col, style=th_style) for col in dataframe.columns]
                ])
            ),
            # Add the table body
            html.Tbody([
                html.Tr([
                    *[html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns]
                ])
                for i in range(min(len(dataframe), max_rows))
            ])
        ]
    )

def generate_graph(df, columns, start_date, end_date):
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'title': column, 'overlaying': 'y', 'side': 'right', 'position': i * 0.1})
    
    # Define the data and layout of the figure
    data = [go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column) for column in filtered_df.columns]
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date')
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    
    return fig


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Defining the layout of the dashboard, with the respective tabs and their descriptions
app.layout = html.Div(style={'backgroundColor': 'white'}, children=[
    html.H1('Solar Power Production in Portugal Forecast Tool'),
    html.P('This dashboard allows for the user to create his own forecast model for Solar Power Production in Portugal, with the best selection of features that he wishes and for the period of time that he desires in the current year of 2024. This forecast can be done: day ahead, week ahead, month ahead, for the first 3 months of the year, etc., and is developed based on the data collected for the years 2022 and 2023 in Portugal. The user also has access to our best results for forecast for day ahead and for the current year of 2024.'),
    html.Div(id='df_total', children=df_total.to_json(orient='split'), style={'display': 'none'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Introduction', value='tab-1', children=[
            html.Div([
                html.H2("Introduction"),
                html.H6('Over the last couple of years we are observing a severe increase of renewable energy sources,and with that it appears the need to forecast for a better management of the resources. For that reason, we decided to create a forecast tool for solar power production for the country of Portugal that has experienced a significant increase in its production on the last couple of years. This dashboard allows for the user to create his own forecast model with the best selection of features that he wishes and for the period of time that he desires in the current year of 2024. This forecast can be done: day ahead, week ahead, month ahead, for the first 3 months of the year, etc., and is developed based on the data collected for the years 2022 and 2023 in Portugal. The user also has access to our best results for forecast for day ahead and for the current year of 2024.'),
                html.H6('This project was done by Catarina Henriques, nº 95997, and João Guerreiro, nº 96407.'),
                html.H6('We found it hard to split up the project because everything was so intertwined. So, we figured we would get a better project if we both looked at the data and worked on the dashboard together, bouncing ideas off each other. So, we ended up teaming up on every part of the project, making sure we both had a say and shared the work equally.'),
                html.H6('We are very proud of our work and we hope the professor enjoys it as well!'),
            ])
        ]),
        dcc.Tab(label='Data 2022/2023', value='tab-2', children=[
            html.Div([
                html.H2("Data 2022/2023"),
                html.P('Check graphically the raw data relative to Solar Power Production, weather and climate for the years 2022 and 2023. Select as many features as you want and adjust the date range as needed.'),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in training_data.columns],
                    value=[training_data.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=training_data.index.min(),
                    max_date_allowed=training_data.index.max(),
                    start_date=training_data.index.min(),
                    end_date=training_data.index.max()
                ),
                dcc.Graph(id='graph'),
            ])
        ]),
        dcc.Tab(label='Data 2024', value='tab-3', children=[
            html.Div([
                html.H2("Data 2024"),
                html.P('Check graphically the raw data Solar Power Production, weather and climate for the year 2024, so far. Select the variables you want to plot and adjust the date range as needed.'),
                dcc.Dropdown(
                id='column-dropdown-2024',
                options=[{'label': i, 'value': i} for i in df_2024_visualize.columns],
                value=[df_2024_visualize.columns[0]],
                multi=True
                ),
                dcc.DatePickerRange(
                id='date-picker-1',
                min_date_allowed=df_2024_visualize.index.min(),
                max_date_allowed=df_2024_visualize.index.max(),
                start_date=df_2024_visualize.index.min(),
                end_date=df_2024_visualize.index.max()
                ),
                dcc.Graph(id='graph-2024'),
            ])
        ]),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-4', children=[
            html.Div([
                html.H2("Exploratory Data Analysis"),
                html.P('Here you have two types of graphical analysis to visualize your data. The first option is a scatter plot where you can select two features and check the relationship between them, which can be a powerful tool for feature selection. The second option is a box plot where you can select only one feature and see the distribution of their values, which is a powerful tool to check for possible outliers.'),
                dcc.Dropdown(
                    id='feature1',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=training_data.columns[0]
                ),
                dcc.Dropdown(
                    id='feature2',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=training_data.columns[1]
                ),
                dcc.Graph(id='scatter-plot'),
                dcc.Dropdown(
                    id='feature-boxplot',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=training_data.columns[0]
                ),
                dcc.Graph(id='box-plot')
            ])
        ]),
        dcc.Tab(label='Correlation Matrix', value='tab-5', children=[
            html.Div([
                html.H2("Correlation Matrix"),
                html.P('Since we noticed there are a lot of similar features on the initial dataset, we decided to add another form of exploratory data analysis, which is the correlation matrix, where the user can select as many features as he wishes to see how correlated are the features between each other and to check which ones can be eliminated based on their high level of correlation.'),
                dcc.Dropdown(
                    id='correlation-features-dropdown',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=[training_data.columns[0], training_data.columns[1]],
                    multi=True
                ),
                dcc.Graph(id='correlation-heatmap'),
            ])
        ]),
        dcc.Tab(label='Forecasting Parameters', value='tab-6', children=[
            html.Div([
                html.H2("Selection of the Features and Period of Time to Forecast"),
                html.P('In this tab we took the liberty to add other features that by our exploratory data analysis we considered that can be important features for an improved forecast. The added features to the original dataset are: Solar Power 1h and Solar Power 2h (Solar Power Production 1 hour and 2 hours before, respectively), Hour, Month and Hot Seasons (corresponding to seasons Spring, Summer and August). Select the features that you wish to use for your model. You can also select the period of time that you wish to forecast for the year of 2024! When it is done, remember to confirm your selection!'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df_meteo_complete.columns],
                    value=[df_meteo_complete.columns[0]],
                    multi=True
                ),
                html.Div(id='feature-table-div'),
                dcc.DatePickerRange(
                id='date-picker-2024',
                min_date_allowed=df_2024.index.min(),
                max_date_allowed=df_2024.index.max(),
                start_date=df_2024.index.min(),
                end_date=df_2024.index.max()
                ),
                html.Button('Confirm your selection', id='split-button'),
                html.Div(id='split-values'),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="y-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-2024-values", style=white_text_style)
                ]),
            ])
        ]),
        dcc.Tab(label='Models', value='tab-7', children=[
            html.Div([
                html.H2("Models"),
                html.P('Select the type of model you want to train with the features and time period for forecast that you previously selected and press the button to train your model. If you dont select anything in the tab Forecasting Parameters, it is impossible to train the model, so try again.'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Auto Regressive', 'value': 'auto_regressive'},
                        {'label': 'Linear Regression', 'value': 'linear'},
                        {'label': 'Random Forests', 'value': 'random_forests'},
                        {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                        {'label': 'Decision Tree Regressor', 'value': 'decision_trees'},
                        {'label': 'Gradient Boosting', 'value': 'gradient_boosting'}
                    ],
                    value='linear'
                ),
                html.Button('Train Model', id='train-model-button'),
            ]),
            html.Div([
                html.H2(""),
                dcc.Loading(
                    id="loading-1",
                    children=[html.Div([dcc.Graph(id="lr-graph")])]
                )
            ]),
        ]),
        dcc.Tab(label='Results of Forecast', value='tab-8', children=[
            html.Div([
                html.H2('Results of Forecast'),
                html.P('Here you can find by default the values of Solar Power Production for 2024. If you wish to check the results of your predictions vs. the actual real values and the performance metrics, press the button. It is worth mentioning that you need to do the steps for Forecasting Parameters and Models to actually have predictions.'),
                dcc.Graph(id='time-series-plot', figure=fig2),
                dcc.Graph(id='scatter-plot-real-predicted'),
                html.Button('Run', id='button_model'),
                html.Div(id='model-performance-table'),
            ])
        ]),
        dcc.Tab(label='Our Best Results', value='tab-9', children=[
            html.Div([
                html.H2('Our Best Results'),
                html.P('In this tab if you click on the button, we show you our best results for the forecast of Solar Power Production and the corresponding performance metrics for the year 2024 and for the day ahead, that we considered to be the first day of the year of 2024 (1st of January). The selected features to obtain this results were Solar Power 1h and 2h, Direct Radiation, Relative Humidity and Temperature and using the model Bootstrapping.'),
                dcc.Graph(id='final-results-plot-2024'),
                html.Div(id='final-results-table-2024'),
                dcc.Graph(id='ahead-results-plot-2024'),
                html.Div(id='ahead-results-table-2024'),
                html.Button('Show', id='show_model')
            ])
        ]),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('graph', 'figure'),
              Input('column-dropdown', 'value'),
              Input('date-picker', 'start_date'),
              Input('date-picker', 'end_date')
)

def update_figure(columns, start_date, end_date):
    
    filtered_df = training_data.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': {'text': ', '.join(columns)}, 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig

@app.callback(Output('graph-2024', 'figure'),
              Input('column-dropdown-2024', 'value'),
              Input('date-picker-1', 'start_date'),
              Input('date-picker-1', 'end_date')
)

def update_figure_2024(columns, start_date, end_date):
    
    filtered_df = df_2024_visualize.loc[start_date:end_date, columns]
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': 'Data 2024', 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('feature1', 'value'),
    Input('feature2', 'value')
)

def update_scatter_plot(feature1, feature2):
    fig = px.scatter(df_total, x=feature1, y=feature2, title=f'{feature1} vs {feature2}')
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    Input('feature-boxplot', 'value')
)

def update_box_plot(feature_boxplot):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_total[feature_boxplot], name=feature_boxplot))
    fig.update_layout(title=f"Box Plot for {feature_boxplot}", title_x=0.5)
    return fig

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('correlation-features-dropdown', 'value')
)

def update_correlation_heatmap(selected_features):
    # Filter the DataFrame based on selected features
    selected_df = training_data[selected_features]
    
    # Compute the correlation matrix
    correlation_matrix = selected_df.corr()
    
    # Generate the heat map
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis'))
    
    # Update layout
    heatmap_fig.update_layout(title='Correlation Matrix Heatmap')
    
    return heatmap_fig

@app.callback(
    Output('feature-table-div', 'children'),
    Input('feature-dropdown', 'value')
)

def update_feature_table(selected_features):
    if selected_features:
        global df_model
        df_model = df_meteo_complete[selected_features]
        table = generate_table(df_model)
        return table
    else:
        return html.Div()
    
@app.callback(
    Output('x-values', 'children'),
    Output('y-values', 'children'),
    Output('x-2024-values', 'children'),
    Input('feature-dropdown', 'value'),
    Input('date-picker-2024', 'start_date'),
    Input('date-picker-2024', 'end_date')
)
def update_x_y(selected_features, start_date, end_date):
    global X, Y, X_2024, num_intervals, Y_2024
    if selected_features:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Calculate num_intervals
        num_intervals = int((end_date - start_date).total_seconds() / 3600) + 1

        # Concatenate selected features with the target variable
        selected_features_with_target = selected_features + ['Solar Power Aggregated [MW]']
        
        # Filter the DataFrame based on selected features
        df_selected_features = training_data_complete[selected_features_with_target]
        
        # Split the data into features (X) and target variable (Y)
        X = df_selected_features.drop(columns=['Solar Power Aggregated [MW]']).values
        Y = df_selected_features['Solar Power Aggregated [MW]'].values 
        # Extract features for the year 2024
        X_2024 = df_meteo_2024[selected_features].loc[start_date:end_date].values
        Y_2024 = df_2024['Solar Power Aggregated [MW]'].loc[start_date:end_date].values
        return str(X), str(Y), str(X_2024)
    else:
        return "", "", ""


@app.callback(
    Output('split-values', 'children'),
    Input('split-button', 'n_clicks')
)

def generate_train_test_split(n_clicks):
    global X_train, X_test, y_train, y_test
    if n_clicks:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = num_intervals, shuffle= False)
        return 'Done! Continue to the next tab if you wish!'
    else:
        return ""
    
#Define global variables
y_pred_list = []
y_pred2024 = []

@app.callback(
    Output('lr-graph', 'figure'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)

def train_and_predict(n_clicks, model_type):
    global y_pred_list, y_pred2024, y_pred # access global variable
    
    if n_clicks is None:
        return dash.no_update 
    else:

        if model_type == 'linear':
            # Linear Regression
            model = LinearRegression()
        elif model_type == 'random_forests':
            # Random Forests
            parameters = {'bootstrap': True,
                          'min_samples_leaf': 3,
                          'n_estimators': 200, 
                          'min_samples_split': 15,
                          'max_features': 'sqrt',
                          'max_depth': 20,
                          'max_leaf_nodes': None}
            model = RandomForestRegressor(**parameters)
        elif model_type == 'bootstrapping':
            # Bootstrapping
            model = BaggingRegressor()
        elif model_type == 'decision_trees':
            # Decision Trees
            model = DecisionTreeRegressor()
        elif model_type == 'auto_regressive':
            # Auto-Regressive (AR) Model
            model = AutoReg(endog=y_train, lags=5)
        elif model_type == 'gradient_boosting':
            # Gradient Boosting
            model = GradientBoostingRegressor()
        
        if model_type == 'auto_regressive':
            ar_model = model.fit()
            # Save the trained model
            with open('ar_model.pkl', 'wb') as file:
                pickle.dump(ar_model, file)
                file.close()

            # Make predictions
            y_pred = ar_model.predict(start=len(y_train), end=len(y_train)+len(X_test)-1, dynamic=False)
            y_pred_list.append(y_pred)
            y_pred2024 = ar_model.predict(start=len(Y), end=len(Y)+len(X_2024)-1, dynamic=False)

            # Generate scatter plot of predicted vs actual values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title=f'AutoRegressive Predictions')
        else:
            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
        
            y_pred2024 = model.predict(X_2024)
        
            # Generate scatter plot of predicted vs actual values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title=f'{model_type.capitalize()} Predictions')
        
        return fig

@app.callback(
    Output('time-series-plot', 'figure'),
    Output('scatter-plot-real-predicted', 'figure'),
    Output('model-performance-table', 'children'),
    Input('button_model', 'n_clicks')
)

def run_model(n_clicks):
    global y_pred2024,Y_2024
    if n_clicks is None:
        raise PreventUpdate
    else:
        #Plot of Real vs. Predicted Power - time series
       
        fig = go.Figure(layout=go.Layout(title='Real vs Predicted Solar Power Production'))
        fig.add_scatter(x = df_2024.index, y = Y_2024, name='Real Power')
        fig.add_scatter(x = df_2024.index, y = y_pred2024, name='Predicted Power') 

        #Plot of Real vs. Predicted Power - scatter plot
        scatter_plot_fig = go.Figure()
        scatter_plot_fig.add_trace(go.Scatter(x=Y_2024, y=y_pred2024, mode='markers'))
        scatter_plot_fig.update_layout(
        title='Real vs Predicted Solar Power Production',
        xaxis_title='Real Power',
        yaxis_title='Predicted Power'
        )

        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(Y_2024, y_pred2024)
        MBE = np.mean(Y_2024 - y_pred2024)
        MSE = metrics.mean_squared_error(Y_2024, y_pred2024)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(Y_2024)
        nmbe = MBE / np.mean(Y_2024)

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvRMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
    return fig, scatter_plot_fig, table

@app.callback(
    Output('final-results-plot-2024', 'figure'),
    Output('final-results-table-2024', 'children'),
    Output('ahead-results-plot-2024', 'figure'),
    Output('ahead-results-table-2024', 'children'),
    Input('show_model', 'n_clicks')
)
def display_final_results(n_clicks):
    global x_2024, y_2024, x_day_ahead, y_day_ahead
    if n_clicks is None:
        raise PreventUpdate
    else:
        # Calculate x_2024 and y_2024 based on user input
        x_2024 = df_meteo_2024[['Temperature (°C)', 'Relative Humidity (%)', 'Direct Radiation (W/m²)', 'Solar Power-1h','Solar Power-2h']].values
        y_2024 = df_2024['Solar Power Aggregated [MW]'].values
        
        # Plot the real vs. predicted values
        with open('BT_model.pkl', 'rb') as file:
            BT_model = pickle.load(file)
        
        y_pred_final = BT_model.predict(x_2024)
        fig_final = go.Figure(layout=go.Layout(title='Real vs Predicted Solar Power Production for the year 2024'))
        fig_final.add_scatter(x = df_2024.index, y = y_2024, name='Real Power')
        fig_final.add_scatter(x = df_2024.index, y = y_pred_final, name='Predicted Power') 
        
        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(y_2024, y_pred_final)
        MBE = np.mean(y_2024 - y_pred_final)
        MSE = metrics.mean_squared_error(y_2024, y_pred_final)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(y_2024)
        nmbe = MBE / np.mean(y_2024)

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvRMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table_final = generate_table(df_metrics)

        #x_day_ahead = x_2024.loc['2024-01-02':'2024-01-03']
        x_2024_df = pd.DataFrame(x_2024, index=df_meteo_2024.index, columns=['Temperature (°C)', 'Relative Humidity (%)', 'Direct Radiation (W/m²)', 'Solar Power-1h', 'Solar Power-2h'])
        x_day_ahead = x_2024_df.loc['2024-01-01':'2024-01-02T00:00:00'].values
        y_day_ahead = df_2024['Solar Power Aggregated [MW]'].loc['2024-01-01':'2024-01-02T00:00:00'].values

        y_pred_ahead = BT_model.predict(x_day_ahead)
        fig_ahead = go.Figure(layout=go.Layout(title='Real vs Predicted Solar Power Production Day Ahead'))
        fig_ahead.add_scatter(x = df_2024.index, y = y_day_ahead, name='Real Power')
        fig_ahead.add_scatter(x = df_2024.index, y = y_pred_ahead, name='Predicted Power') 
        
        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(y_day_ahead, y_pred_ahead)
        MBE = np.mean(y_day_ahead - y_day_ahead)
        MSE = metrics.mean_squared_error(y_day_ahead, y_pred_ahead)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(y_day_ahead)
        nmbe = MBE / np.mean(y_day_ahead)

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvRMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table_ahead = generate_table(df_metrics)


    return fig_final, table_final, fig_ahead, table_ahead

    
if __name__ == '__main__':
    app.run_server()