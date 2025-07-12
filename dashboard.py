# dashboard.py - Simple Streamlit Dashboard
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

st.set_page_config(
    page_title="Alpha Engine Dashboard",
    page_icon="🚀",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load data from SQLite database"""
    try:
        conn = sqlite3.connect('alpha_engine.db')
        
        # Load signals
        signals_query = """
        SELECT * FROM signals 
        WHERE signal_date >= date('now', '-30 days')
        ORDER BY signal_date DESC
        """
        signals_df = pd.read_sql_query(signals_query, conn)
        
        # Load insider trades
        trades_query = """
        SELECT * FROM insider_trades 
        WHERE transaction_date >= date('now', '-30 days')
        ORDER BY transaction_date DESC
        """
        trades_df = pd.read_sql_query(trades_query, conn)
        
        conn.close()
        return signals_df, trades_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    st.title("🚀 Alpha Engine Dashboard")
    st.markdown("AI-Powered Insider Trading Signal Generator")
    
    # Load data
    signals_df, trades_df = load_data()
    
    if signals_df.empty and trades_df.empty:
        st.warning("No data available. Please run the Alpha Engine pipeline first.")
        return
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    if not signals_df.empty:
        # Signal strength filter
        strength_options = signals_df['signal_strength'].unique() if 'signal_strength' in signals_df.columns else []
        selected_strength = st.sidebar.multiselect("Signal Strength", strength_options, default=strength_options)
        
        # Action filter
        action_options = signals_df['recommended_action'].unique() if 'recommended_action' in signals_df.columns else []
        selected_actions = st.sidebar.multiselect("Recommended Action", action_options, default=action_options)
        
        # Filter signals
        if selected_strength and selected_actions:
            filtered_signals = signals_df[
                (signals_df['signal_strength'].isin(selected_strength)) &
                (signals_df['recommended_action'].isin(selected_actions))
            ]
        else:
            filtered_signals = signals_df
    else:
        filtered_signals = pd.DataFrame()
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals (30d)", len(signals_df))
    
    with col2:
        buy_signals = len(signals_df[signals_df['recommended_action'] == 'BUY']) if not signals_df.empty else 0
        st.metric("Buy Signals", buy_signals)
    
    with col3:
        strong_signals = len(signals_df[signals_df['signal_strength'] == 'Strong']) if not signals_df.empty else 0
        st.metric("Strong Signals", strong_signals)
    
    with col4:
        total_trades = len(trades_df)
        st.metric("Insider Trades (30d)", total_trades)
    
    # Charts section
    st.header("📊 Signal Analysis")
    
    if not filtered_signals.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal strength distribution
            fig_strength = px.pie(
                filtered_signals, 
                names='signal_strength', 
                title="Signal Strength Distribution"
            )
            st.plotly_chart(fig_strength, use_container_width=True)
        
        with col2:
            # Signals over time
            if 'signal_date' in filtered_signals.columns:
                signals_by_date = filtered_signals.groupby('signal_date').size().reset_index(name='count')
                fig_timeline = px.line(
                    signals_by_date, 
                    x='signal_date', 
                    y='count',
                    title="Signals Over Time"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Recent signals table
    st.header("🎯 Recent Signals")
    
    if not filtered_signals.empty:
        # Display signals table
        display_cols = ['ticker', 'company_name', 'signal_date', 'signal_probability', 
                       'signal_strength', 'recommended_action']
        available_cols = [col for col in display_cols if col in filtered_signals.columns]
        
        st.dataframe(
            filtered_signals[available_cols].head(20),
            use_container_width=True
        )
        
        # Download button
        csv = filtered_signals.to_csv(index=False)
        st.download_button(
            label="Download Signals as CSV",
            data=csv,
            file_name=f"alpha_signals_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No signals match the selected filters.")
    
    # Recent insider trades
    st.header("📈 Recent Insider Trades")
    
    if not trades_df.empty:
        display_cols = ['ticker', 'company_name', 'transaction_date', 'insider_name',
                       'insider_title', 'transaction_code', 'shares', 'transaction_value']
        available_cols = [col for col in display_cols if col in trades_df.columns]
        
        st.dataframe(
            trades_df[available_cols].head(20),
            use_container_width=True
        )
    
    # Stock price chart for selected ticker
    st.header("📊 Stock Price Analysis")
    
    if not signals_df.empty and 'ticker' in signals_df.columns:
        tickers = signals_df['ticker'].dropna().unique()
        selected_ticker = st.selectbox("Select ticker for price analysis:", tickers)
        
        if selected_ticker:
            try:
                # Get stock data
                stock = yf.Ticker(selected_ticker)
                hist = stock.history(period="3mo")
                
                if not hist.empty:
                    # Create price chart
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add insider trade markers
                    ticker_trades = trades_df[trades_df['ticker'] == selected_ticker]
                    if not ticker_trades.empty:
                        for _, trade in ticker_trades.iterrows():
                            trade_date = pd.to_datetime(trade['transaction_date'])
                            if trade_date in hist.index:
                                color = 'green' if trade['transaction_code'] == 'P' else 'red'
                                fig.add_trace(go.Scatter(
                                    x=[trade_date],
                                    y=[hist.loc[trade_date, 'Close']],
                                    mode='markers',
                                    marker=dict(size=10, color=color),
                                    name=f"Insider {trade['transaction_code']}",
                                    showlegend=False
                                ))
                    
                    fig.update_layout(
                        title=f"{selected_ticker} Price Chart with Insider Trades",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show company info
                    info = stock.info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${hist['Close'][-1]:.2f}")
                    with col2:
                        st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
                    with col3:
                        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                
            except Exception as e:
                st.error(f"Error loading stock data: {e}")

if __name__ == "__main__":
    main()
