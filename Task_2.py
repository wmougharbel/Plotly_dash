import dash
from dash import dcc, html, callback, Output, Input
import pandas as pd
from datetime import datetime
import plotly.express as px
import yfinance as yf

# Define stock symbols
stock_symbols = ['AAPL', 'AMGN', 'AXP', 'BA', 'BAC', 'BEN', 'BRK-B', 'CAT', 'CSCO', 'CVX',
				'CXO', 'DD', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FB', 'FDX', 'GE', 
				'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 
				'JPM', 'KO', 'LLY', 'LMT', 'MA', 'MCD', 'MMM', 'MRK', 'MSFT', 'MSI', 
				'NEE', 'NEM', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'OXY', 'PEP', 'PG', 'PM', 
				'QCOM', 'RTX', 'SBUX', 'SLB', 'SO', 'SPG', 'T', 'TRV', 'UNH', 'UTX', 
				'V', 'VZ', 'WMT', 'XOM', 'XRAY']

# Define date range
start_date = int(datetime(2024, 6, 1).timestamp())
end_date = int(datetime(2024, 12, 31).timestamp())

def calculate_rsi(data, window=14):
	delta = data['Close'].diff()
	gain = (delta.where(delta > 0.0, 0)).rolling(window=window).mean()
	loss = (-delta.where(delta < 0.0, 0)).rolling(window=window).mean()
	rs = gain / loss
	rsi = 100 - (100 / (1 + rs))
	return rsi

def fetch_data_from_api(start_date, end_date):
	dataframes = {}
	for symbol in stock_symbols:
		print(f"Fetching data for {symbol}...")
		ticker = yf.Ticker(symbol)
		df = ticker.history(start=start_date, end=end_date)
		if not df.empty:
			df['Date'] = df.index.strftime('%d-%m-%Y').tolist()
			df['Symbol'] = symbol
			df['RSI'] = calculate_rsi(df)
			dataframes[symbol] = df

	combined_df = pd.concat(dataframes.values(), ignore_index=True)
	combined_df.to_csv('stock_data.csv', index=False)

def init_app():
	# Create the layout for Task 2
	layout = html.Div([
		html.H2("Stock Price Line Chart", style={'text-align': 'center'}),

		dcc.Dropdown(
			id='stock-dropdown',
			options=[{'label': symbol, 'value': symbol} for symbol in stock_symbols],
			value='AAPL',
			multi=True
		),

		dcc.Checklist(
			id='rsi-checklist',
			options=[{'label': 'Show RSI', 'value': 'RSI'}],
			value=[],
			inline=True
		),

		dcc.Graph(id='line-chart')
	])
	return layout

@callback(
	Output('line-chart', 'figure'),
	[Input('stock-dropdown', 'value'), Input('rsi-checklist', 'value')]
)
def draw_line_chart(selected_stocks, show_rsi):
	try:
		stock_data = pd.read_csv('stock_data.csv')
	except:
		fetch_data_from_api(start_date, end_date)
		stock_data = pd.read_csv('stock_data.csv')

	if stock_data.empty:
		return None

	if isinstance(selected_stocks, str):
		selected_stocks = [selected_stocks]

	stock_data = stock_data[stock_data['Symbol'].isin(selected_stocks)]

	fig = px.line(
		stock_data,
		x='Date',
		y='Close',
		color='Symbol',
		title="Stock Price for Selected Stocks",
		labels={'Date': 'Date', 'Close': 'Price ($)', 'Symbol': 'Stock'}
	)

	if 'RSI' in show_rsi:
		for stock in selected_stocks:
			rsi_data = stock_data[stock_data['Symbol'] == stock]
			fig.add_scatter(
				x = rsi_data['Date'],
				y = rsi_data['RSI'],
				mode = 'lines',
				name = f'{stock} RSI'
			)

	fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)", xaxis_rangeslider_visible=True)

	return fig