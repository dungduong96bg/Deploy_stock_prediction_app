import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=120, value=60)
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=20)

# Main content
if symbol:
    # 1. Fetch Data
    st.subheader(f"1. Historical Data for {symbol}")
    
    with st.spinner(f"Fetching data for {symbol}..."):
        # Fetch data from year 2018 to ensure enough data for training
        df = get_stock_data(symbol, start_date="2018-01-01")
    
    if df is not None:
        # Display Data
        st.dataframe(df.tail())
        
        # TradingView Widget
        st.subheader("2. TradingView Chart")
        
        # Mapping for TradingView (assuming Vietnam stocks are on HOSE/HNX, but TradingView symbol format might vary)
        # For simplicity, we try to use the symbol directly or prefix with 'HOSE:'
        tv_symbol = f"HOSE:{symbol}"
        
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
                # Initialize Predictor
                status_text.text("Initializing model...")
                predictor = StockMarketPredictor(lookback=lookback)
                
                # Prepare Data
                status_text.text("Preparing data...")
                X_train, X_test, y_train, y_test, scaled_data = predictor.prepare_data(df)
                
                progress_bar.progress(20)
                
                # Train
                status_text.text(f"Training model for {epochs} epochs... This may take a while.")
                history = predictor.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=32)
                
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
                
                # Plot Results using Plotly for interactivity
                st.subheader("Prediction vs Actual")
                
                # Create a DataFrame for plotting
                # y_test and predictions are scaled values? No, wait.
                # In stock_market_prediction.py:
                # y is from scaled_data.
                # predictions are output of model.predict(X_test).
                # Both are scaled. We need to inverse transform them to get actual prices.
                
                # Inverse Transform
                # The scaler was fitted on ['Open', 'High', 'Low', 'Close', 'Volume']
                # We need to inverse transform carefully.
                
                # Let's look at how we can inverse transform.
                # The scaler is in predictor.scaler
                # It expects 5 features.
                # We have y_test (1 feature: Close) and predictions (1 feature: Close).
                # We need to construct a dummy array with 5 features to inverse transform, 
                # or just manually unscale if we know the min/max of the Close column.
                # But MinMaxScaler scales each feature independently.
                
                # Let's access the scaler's min and scale for the Close column.
                # features = ['Open', 'High', 'Low', 'Close', 'Volume']
                # Close is at index 3.
                
                scaler = predictor.scaler
                close_min = scaler.data_min_[3]
                close_range = scaler.data_range_[3]
                
                y_test_actual = y_test * close_range + close_min
                predictions_actual = predictions.flatten() * close_range + close_min
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test_actual, mode='lines', name='Actual Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=predictions_actual, mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
                
                fig.update_layout(
                    title=f"{symbol} Price Prediction",
                    xaxis_title="Time Steps (Test Set)",
                    yaxis_title="Price (VND)",
                    template="plotly_white"
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
                
    else:
        st.warning(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
else:
    st.info("Please enter a stock symbol to start.")
