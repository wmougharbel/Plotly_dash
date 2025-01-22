import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from Task_1 import init_app as init_task_1
from Task_2 import init_app as init_task_2
from Task_3 import init_app as init_task_3

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
	dcc.Location(id='url', refresh=False),
	html.Div([
		dcc.Link('Task 1', href='/task1', style={'margin-right': '20px'}),
		dcc.Link('Task 2', href='/task2', style={'margin-right': '20px'}),
		dcc.Link('Task 3', href='/task3'),
	]),
	html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
			Input('url', 'pathname'))
def display_page(pathname):
	if pathname == '/task1':
		return init_task_1()
	elif pathname == '/task2':
		return init_task_2()
	elif pathname == '/task3':
		return init_task_3()
	else:
		return init_task_1()

if __name__ == '__main__':
	app.run_server(debug=True)