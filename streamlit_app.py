import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

# Import local modules
from Get_Data import get_stock_data
from stock_market_prediction import StockMarketPredictor

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide"
)

# Title and Description
st.title("📈 Stock Price Prediction App")
st.markdown("""
This app allows you to input any stock symbol, view its historical data, 
and predict future prices using an **LSTM-DNN Hybrid Model**.
""")

# Sidebar for inputs
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., FPT, VCB)", value="FPT").upper()
exchange = st.sidebar.selectbox("Select Exchange", ["HOSE", "HNX", "UPCOM", "NASDAQ", "NYSE"], index=0)
lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=120, value=60)
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=20)

# Caching data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(symbol):
    return get_stock_data(symbol, start_date="2018-01-01")

# Caching model training
@st.cache_resource
def train_model(symbol, lookback, epochs, _data):
    """
    Train the model and return the predictor object and history.
    Using _data to prevent hashing the dataframe which can be slow.
    We rely on symbol/lookback/epochs to trigger retraining.
    """
    predictor = StockMarketPredictor(lookback=lookback)
    X_train, X_test, y_train, y_test, scaled_data = predictor.prepare_data(_data)
    
    history = predictor.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32)
    return predictor, history, X_test, y_test, scaled_data

# Main content
if symbol:
    # 1. Fetch Data
    st.subheader(f"1. Historical Data for {symbol}")
    
    with st.spinner(f"Fetching data for {symbol}..."):
        df = fetch_data(symbol)
    
    if df is not None:
        # Display Data
        st.dataframe(df.tail())
        
        # TradingView Widget
        st.subheader("2. TradingView Chart")
        
        tv_symbol = f"{exchange}:{symbol}"
        
        html_code = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div id="tradingview_{symbol}"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
          "width": "100%",
          "height": 500,
          "symbol": "{tv_symbol}",
          "interval": "D",
          "timezone": "Asia/Ho_Chi_Minh",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "container_id": "tradingview_{symbol}"
        }}
          );
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        st.components.v1.html(html_code, height=500)
        
        # 3. Model Training & Prediction
        st.subheader("3. AI Prediction (LSTM-DNN)")
        
        if st.button("Train Model & Predict"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Train or Load Model
                status_text.text("Training model... (Cached if parameters unchanged)")
                predictor, history, X_test, y_test, scaled_data = train_model(symbol, lookback, epochs, df)
                
                progress_bar.progress(80)
                
                # Evaluate
                status_text.text("Evaluating model...")
                metrics, predictions = predictor.evaluate(X_test, y_test)
                
                progress_bar.progress(100)
                status_text.text("Prediction complete!")
                
                # Display Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{metrics['R2']:.4f}")
                col2.metric("MAE", f"{metrics['MAE']:.4f}")
                col3.metric("MSE", f"{metrics['MSE']:.4f}")
                col4.metric("RMSE", f"{metrics['RMSE']:.4f}")
                
                # --- Next Day Prediction Logic ---
                # Get the last 'lookback' days from the scaled data to predict the next day
                last_sequence = scaled_data[-lookback:]
                last_sequence = last_sequence.reshape(1, lookback, scaled_data.shape[1])
                
                next_day_prediction_scaled = predictor.model.predict(last_sequence)
                
                # Inverse Transform Logic
                scaler = predictor.scaler
                close_min = scaler.data_min_[3] # Close is at index 3
                close_range = scaler.data_range_[3]
                
                next_day_price = next_day_prediction_scaled[0][0] * close_range + close_min
                
                st.success(f"### 🔮 Predicted Price for Next Trading Day: {next_day_price:,.0f} VND")
                
                # --- Plot Results with Dates ---
                st.subheader("Prediction vs Actual")
                
                # Get dates for the test set
                # The test set is the last 20% of the data (minus lookback adjustment)
                # In prepare_data:
                # X has length: len(scaled_data) - lookback
                # split_idx = int(len(X) * 0.8)
                # X_test starts from split_idx
                
                # So the corresponding dates for y_test start from:
                # lookback + split_idx
                
                total_samples = len(scaled_data) - lookback
                split_idx = int(total_samples * 0.8)
                test_start_idx = lookback + split_idx
                
                # df['Date'] aligns with scaled_data indices
                test_dates = df['Date'].iloc[test_start_idx:].values
                
                # Inverse Transform Test Data
                y_test_actual = y_test * close_range + close_min
                predictions_actual = predictions.flatten() * close_range + close_min
                
                # Ensure lengths match (sometimes off by 1 due to slicing)
                min_len = min(len(test_dates), len(y_test_actual))
                test_dates = test_dates[:min_len]
                y_test_actual = y_test_actual[:min_len]
                predictions_actual = predictions_actual[:min_len]
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, mode='lines', name='Actual Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_dates, y=predictions_actual, mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
                
                fig.update_layout(
                    title=f"{symbol} Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price (VND)",
                    template="plotly_white",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Training History
                st.subheader("Training History")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
                fig_hist.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss'))
                fig_hist.update_layout(title="Model Loss", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during training/prediction: {e}")
                st.error("Please check if the data is sufficient or try a different symbol.")
                import traceback
                st.text(traceback.format_exc())
                
    else:
        st.warning(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
else:
    st.info("Please enter a stock symbol to start.")
