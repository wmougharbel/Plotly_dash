import dash
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from dash import dcc, html, dash_table, callback, Output, Input

def read_csv_file(csv_file, state):
	df = pd.read_csv(csv_file)
	df['When'] = pd.to_datetime(df['When'], errors='coerce', dayfirst=True)
	df = df[df['Sex'] != 'Both sexes']
	state_data = df[['When', state]].dropna()  # Remove missing values
	scaler = MinMaxScaler()
	state_data['Normalized_Rate'] = scaler.fit_transform(state_data[[state]])  # Scale data between 0 and 1 for better clustering
	state_data['Days'] = (state_data['When'] - state_data['When'].min()).dt.days + 1  # Convert to numeric values for linear regression
	return state_data

def linear_regression_prediction(state, state_data, date):
	X = state_data[['Days']]
	Y = state_data[state]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	# Train the model
	model = LinearRegression()
	model.fit(X_train, Y_train)

	y_pred = model.predict(X_test)
	mse = mean_squared_error(Y_test, y_pred)
	rmse = np.sqrt(mse)

	future_date = pd.to_datetime(date, dayfirst=True)
	future_days = (future_date - state_data['When'].min()).days + 1
	future_days_df = pd.DataFrame([[future_days]], columns=['Days'])  # Convert it to a dataframe to silence the warning
	future_pred = model.predict(future_days_df)[0]

	return future_pred, rmse

def kmeans_clustering(state_data):
	X = state_data[['Days', 'Normalized_Rate']]

	# Apply KMeans
	kmeans = KMeans(n_clusters=3, random_state=42)
	state_data['Cluster'] = kmeans.fit_predict(X)

	return state_data

def init_app():
	# Create the layout for Task 3
	layout = html.Div([
		html.H1('Unemployment Rate in Canadian States Prediction and Clustering'),
		html.Label('Select State'),
		dcc.Dropdown(
			id='state-dropdown',
			options=[
				{'label': state, 'value': state}
				for state in ['Alberta', 'BritishColumbia', 'Canada', 'Manitoba', 'NewBrunswick',
							'NewfoundlandAndLabrador', 'NovaScotia', 'Ontario', 'PrinceEdwardIsland',
							'Quebec', 'Saskatchewan']
			],
			value='Alberta',
			style={'width': '50%'}
		),
		html.Label('Select Date:'),
		dcc.DatePickerSingle(
			id='date-picker',
			min_date_allowed='2015-01-06',
			max_date_allowed='2035-12-31',
			initial_visible_month='2025-01',
			date='2025-01-01'
		),
		html.Div(id='prediction-output'),
		html.Div([
			dash_table.DataTable(
				id='prediction-table',
				columns=[
					{'name': 'State', 'id': 'State'},
					{'name': 'Selected Date', 'id': 'Date'},
					{'name': 'Predicted Unemployment Rate (%)', 'id': 'Prediction'},
					{'name': 'Root Mean Squared Error (RMSE)', 'id': 'RMSE'}
				],
				style_table={'width': '70%', 'margin': 'auto'},
				style_cell={'textAlign': 'center'}
			)
		]),
		html.Div([
			html.P('Day 1 = 1/1/1976'),
			dcc.Graph(id='kmeans-graph')
		])
	])
	return layout

@callback(
	[Output('prediction-output', 'children'),
	Output('prediction-table', 'data'),
	Output('kmeans-graph', 'figure')],
	[Input('state-dropdown', 'value'),
	Input('date-picker', 'date')]
)
def update_dashboard(state, date):
	if state is None:
		state = 'Alberta'
	state_data = read_csv_file("unemployment_rate.csv", state)
	future_pred, rmse = linear_regression_prediction (state, state_data, date)
	table_data = [{
		'State': state,
		'Date': date,
		'Prediction': f'{future_pred:.2f}',
		'RMSE': f'{rmse:.2f}'
	}]
	prediction_output = html.P(f'Prediction for {state} on {date}: {future_pred:.2f}%')

	clustered_data = kmeans_clustering(state_data)
	kmeans_fig = px.scatter(clustered_data, x='Days', y='Normalized_Rate', color='Cluster', title=f'KMeans Clustering for {state}')
	return prediction_output, table_data, kmeans_fig, 

# if __name__ == '__main__':
#     app.run_server(debug=True)