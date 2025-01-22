from dash import html, dcc, callback, Output, Input
import plotly.express as px
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

def init_app():
	# Create the layout for Task 1
	layout = html.Div([
		html.H1("Iris Dataset"),

		# Dropdowns for selecting X and Y axes
		html.Div([
			html.Label("X axis"),
			dcc.Dropdown(
				id='x-dropdown',
				options=[{'label': feature, 'value': feature} for feature in df.columns],
				value='sepal length (cm)'
			)
		]),

		html.Div([
			html.Label("Y axis"),
			dcc.Dropdown(
				id='y-dropdown',
				options=[{'label': feature, 'value': feature} for feature in df.columns],
				value='petal length (cm)'
			)
		]),

		dcc.Graph(id='iris-scatter-plot')
	])

	return layout

def create_scatter_plot(user_x, user_y):
	if user_x == user_y:
		user_x = 'sepal length (cm)'
		user_y = 'petal length (cm)'

	fig = px.scatter(
		df,
		x=user_x,
		y=user_y,
		title=f"Iris Dataset Scatter Plot: {user_x} vs {user_y}",
		labels={user_x: user_x, user_y: user_y},
	)
	return fig

# Define the callback function
@callback(
	Output('iris-scatter-plot', 'figure'),
	[
		Input('x-dropdown', 'value'),
		Input('y-dropdown', 'value'),
	]
)
def update_scatter_plot(x, y):
	return create_scatter_plot(x, y)

# The following line is not needed in this file since the app will be run from the main app file
# if __name__ == '__main__':
#     app.run_server(debug=True)