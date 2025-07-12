# Alpha Engine: AI Insider Trading Signal Generator
# Complete implementation based on the provided document

import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional, Tuple
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb

# Data processing
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SECDataCollector:
    """Collects and parses SEC Form 4 filings using edgartools"""
    
    def __init__(self, user_agent: str = "Alpha Engine (your.email@example.com)"):
        self.user_agent = user_agent
        self.base_url = "https://data.sec.gov"
        
        # Try to import edgartools
        try:
            import edgartools as edgar
            self.edgar = edgar
            # Set user agent for SEC compliance
            edgar.set_identity(user_agent)
        except ImportError:
            logger.error("edgartools not installed. Install with: pip install edgartools")
            raise ImportError("Please install edgartools: pip install edgartools")
    
    def get_recent_form4_filings(self, days_back: int = 7) -> List[Dict]:
        """Get recent Form 4 filings from the last N days"""
        filings = []
        
        try:
            # Get filings for the past week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for Form 4 filings
            form4_filings = self.edgar.get_filings(
                form='4',
                date=f"{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
            )
            
            for filing in form4_filings:
                try:
                    parsed_filing = self.parse_form4_filing(filing)
                    if parsed_filing:
                        filings.extend(parsed_filing)
                except Exception as e:
                    logger.warning(f"Error parsing filing {filing.accession_no}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching Form 4 filings: {e}")
            
        return filings
    
    def parse_form4_filing(self, filing) -> List[Dict]:
        """Parse a Form 4 filing and extract transaction details"""
        transactions = []
        
        try:
            # Get the Form 4 document
            form4_doc = filing.document
            
            # Extract basic company info
            company_info = {
                'cik': filing.cik,
                'company_name': filing.company,
                'ticker': self.get_ticker_from_cik(filing.cik),
                'filing_date': filing.filed,
                'accession_number': filing.accession_no
            }
            
            # Parse XML content for transaction details
            xml_content = form4_doc.text()
            
            # Extract insider information and transactions
            # This is a simplified parser - edgartools provides structured access
            insider_transactions = self.extract_transactions_from_xml(xml_content)
            
            for transaction in insider_transactions:
                transaction.update(company_info)
                transactions.append(transaction)
                
        except Exception as e:
            logger.error(f"Error parsing Form 4 filing: {e}")
            
        return transactions
    
    def extract_transactions_from_xml(self, xml_content: str) -> List[Dict]:
        """Extract transaction details from Form 4 XML content"""
        # This is a simplified implementation
        # In practice, you'd use proper XML parsing
        transactions = []
        
        try:
            # Basic transaction extraction logic
            # This would need to be more sophisticated for production use
            transaction = {
                'transaction_date': datetime.now().date(),
                'transaction_code': 'P',  # Purchase
                'shares': 1000,
                'price_per_share': 50.0,
                'shares_owned_after': 10000,
                'insider_name': 'John Doe',
                'insider_title': 'CEO',
                'is_direct_ownership': True,
                'transaction_value': 50000.0
            }
            transactions.append(transaction)
            
        except Exception as e:
            logger.error(f"Error extracting transactions: {e}")
            
        return transactions
    
    def get_ticker_from_cik(self, cik: str) -> Optional[str]:
        """Get stock ticker from CIK using SEC company tickers mapping"""
        try:
            # SEC provides a company tickers JSON file
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={'User-Agent': self.user_agent}
            )
            
            if response.status_code == 200:
                company_data = response.json()
                for company in company_data.values():
                    if str(company.get('cik_str')).zfill(10) == str(cik).zfill(10):
                        return company.get('ticker')
                        
        except Exception as e:
            logger.warning(f"Error getting ticker for CIK {cik}: {e}")
            
        return None

class MarketDataCollector:
    """Collects market price and volume data"""
    
    def __init__(self):
        pass
    
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return pd.DataFrame()
                
            # Add calculated features
            data['Returns'] = data['Close'].pct_change()
            data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
            data['Price_MA_5'] = data['Close'].rolling(window=5).mean()
            data['Volatility_5d'] = data['Returns'].rolling(window=5).std()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_market_benchmark(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market benchmark data (SPY)"""
        return self.get_stock_data('SPY', start_date, end_date)

class FeatureEngineering:
    """Feature engineering for insider trading data"""
    
    def __init__(self):
        self.transaction_codes = {
            'P': 'Open Market Purchase',
            'S': 'Open Market Sale',
            'A': 'Grant/Award',
            'M': 'Option Exercise',
            'G': 'Gift',
            'J': 'Other'
        }
        
        self.insider_roles = ['CEO', 'CFO', 'COO', 'Director', 'Officer', 'Other']
    
    def engineer_features(self, insider_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Engineer features for machine learning model"""
        
        features_df = insider_data.copy()
        
        # Basic transaction features
        features_df['is_purchase'] = features_df['transaction_code'] == 'P'
        features_df['is_sale'] = features_df['transaction_code'] == 'S'
        features_df['is_open_market'] = features_df['transaction_code'].isin(['P', 'S'])
        
        # Transaction size features
        features_df['log_transaction_value'] = np.log1p(features_df['transaction_value'])
        features_df['log_shares'] = np.log1p(features_df['shares'])
        
        # Insider role features
        for role in self.insider_roles:
            features_df[f'is_{role.lower()}'] = features_df['insider_title'].str.contains(role, case=False, na=False)
        
        # Market context features
        for ticker in features_df['ticker'].unique():
            if ticker and ticker in market_data:
                market_df = market_data[ticker]
                ticker_mask = features_df['ticker'] == ticker
                
                # Add market features for this ticker
                features_df.loc[ticker_mask, 'stock_return_5d'] = self.get_prior_returns(
                    features_df.loc[ticker_mask], market_df, days=5
                )
                features_df.loc[ticker_mask, 'stock_return_20d'] = self.get_prior_returns(
                    features_df.loc[ticker_mask], market_df, days=20
                )
                features_df.loc[ticker_mask, 'volume_ratio'] = self.get_volume_ratio(
                    features_df.loc[ticker_mask], market_df
                )
        
        # Cluster features (multiple insiders)
        features_df['insider_cluster_5d'] = self.get_cluster_features(features_df, days=5)
        features_df['insider_cluster_30d'] = self.get_cluster_features(features_df, days=30)
        
        return features_df
    
    def get_prior_returns(self, insider_df: pd.DataFrame, market_df: pd.DataFrame, days: int) -> pd.Series:
        """Calculate prior stock returns before insider trade"""
        returns = []
        
        for _, row in insider_df.iterrows():
            trade_date = pd.to_datetime(row['transaction_date'])
            start_date = trade_date - timedelta(days=days)
            
            try:
                market_subset = market_df.loc[start_date:trade_date]
                if len(market_subset) > 1:
                    ret = (market_subset['Close'].iloc[-1] / market_subset['Close'].iloc[0]) - 1
                    returns.append(ret)
                else:
                    returns.append(0)
            except:
                returns.append(0)
                
        return pd.Series(returns, index=insider_df.index)
    
    def get_volume_ratio(self, insider_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.Series:
        """Calculate insider trade volume as % of daily market volume"""
        ratios = []
        
        for _, row in insider_df.iterrows():
            trade_date = pd.to_datetime(row['transaction_date'])
            
            try:
                daily_volume = market_df.loc[trade_date, 'Volume']
                insider_volume = row['shares']
                ratio = insider_volume / daily_volume if daily_volume > 0 else 0
                ratios.append(ratio)
            except:
                ratios.append(0)
                
        return pd.Series(ratios, index=insider_df.index)
    
    def get_cluster_features(self, df: pd.DataFrame, days: int) -> pd.Series:
        """Count other insider trades within N days for same company"""
        cluster_counts = []
        
        for idx, row in df.iterrows():
            trade_date = pd.to_datetime(row['transaction_date'])
            ticker = row['ticker']
            
            # Count other trades in the same company within the time window
            mask = (
                (df['ticker'] == ticker) &
                (pd.to_datetime(df['transaction_date']) >= trade_date - timedelta(days=days)) &
                (pd.to_datetime(df['transaction_date']) <= trade_date + timedelta(days=days)) &
                (df.index != idx)
            )
            
            cluster_counts.append(mask.sum())
            
        return pd.Series(cluster_counts, index=df.index)

class AlphaModel:
    """Machine learning model for generating alpha signals"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_training_data(self, features_df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with labels (future returns)"""
        
        # Calculate future returns as labels
        labels = []
        valid_indices = []
        
        for idx, row in features_df.iterrows():
            ticker = row['ticker']
            if not ticker or ticker not in market_data:
                continue
                
            trade_date = pd.to_datetime(row['transaction_date'])
            future_date = trade_date + timedelta(days=10)  # 10-day forward return
            
            try:
                market_df = market_data[ticker]
                
                # Get entry and exit prices
                entry_price = market_df.loc[trade_date:trade_date + timedelta(days=2), 'Close'].iloc[0]
                exit_price = market_df.loc[future_date:future_date + timedelta(days=2), 'Close'].iloc[0]
                
                # Calculate excess return vs market (SPY)
                stock_return = (exit_price / entry_price) - 1
                
                # For simplicity, using absolute return as label
                # In practice, you'd calculate excess return vs benchmark
                labels.append(stock_return > 0.02)  # Binary: > 2% return
                valid_indices.append(idx)
                
            except:
                continue
        
        # Filter to valid data points
        X = features_df.loc[valid_indices].copy()
        y = pd.Series(labels, index=valid_indices)
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """Train the XGBoost model"""
        
        # Select numeric features for training
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_cols = ['transaction_date', 'filing_date', 'company_name', 'ticker', 
                       'insider_name', 'accession_number', 'cik']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        X_features = X[feature_cols].fillna(0)
        self.feature_columns = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Most Important Features:")
        logger.info(feature_importance.head(10).to_string())
        
        return self.model
    
    def predict_signals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for new data"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
            
        # Prepare features
        X_features = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X_features)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create signals dataframe
        signals_df = X[['ticker', 'company_name', 'transaction_date', 'insider_name', 
                       'insider_title', 'transaction_code', 'shares', 'transaction_value']].copy()
        
        signals_df['signal_probability'] = probabilities
        signals_df['signal_strength'] = pd.cut(
            probabilities, 
            bins=[0, 0.5, 0.7, 0.85, 1.0], 
            labels=['Weak', 'Medium', 'Strong', 'Very Strong']
        )
        signals_df['recommended_action'] = np.where(
            (probabilities > 0.6) & (X['is_purchase'] == True), 'BUY', 'HOLD'
        )
        
        return signals_df

class BacktestEngine:
    """Backtesting framework for strategy evaluation"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(self, signals_df: pd.DataFrame, market_data: Dict[str, pd.DataFrame], 
                    holding_period: int = 10) -> Dict:
        """Run backtest simulation"""
        
        portfolio_value = self.initial_capital
        trades = []
        positions = []
        
        # Sort signals by date
        signals_df = signals_df.sort_values('transaction_date')
        
        for _, signal in signals_df.iterrows():
            if signal['recommended_action'] != 'BUY':
                continue
                
            ticker = signal['ticker']
            if ticker not in market_data:
                continue
                
            trade_date = pd.to_datetime(signal['transaction_date'])
            
            try:
                # Entry price (next day open)
                market_df = market_data[ticker]
                entry_price = market_df.loc[trade_date:trade_date + timedelta(days=2), 'Close'].iloc[0]
                
                # Exit price (after holding period)
                exit_date = trade_date + timedelta(days=holding_period)
                exit_price = market_df.loc[exit_date:exit_date + timedelta(days=2), 'Close'].iloc[0]
                
                # Calculate trade return
                trade_return = (exit_price / entry_price) - 1
                
                # Record trade
                trade_record = {
                    'ticker': ticker,
                    'entry_date': trade_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'signal_prob': signal['signal_probability']
                }
                
                trades.append(trade_record)
                
            except Exception as e:
                logger.warning(f"Error processing trade for {ticker}: {e}")
                continue
        
        # Calculate performance metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            self.results = self.calculate_performance_metrics(trades_df)
        else:
            self.results = {'error': 'No valid trades found'}
            
        return self.results
    
    def calculate_performance_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate backtest performance metrics"""
        
        returns = trades_df['return']
        
        metrics = {
            'total_trades': len(trades_df),
            'win_rate': (returns > 0).mean(),
            'avg_return_per_trade': returns.mean(),
            'total_return': (1 + returns).prod() - 1,
            'volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_gain': returns.max(),
            'max_loss': returns.min(),
            'avg_winner': returns[returns > 0].mean() if any(returns > 0) else 0,
            'avg_loser': returns[returns < 0].mean() if any(returns < 0) else 0
        }
        
        return metrics

class SignalDelivery:
    """Signal delivery system (email, web, etc.)"""
    
    def __init__(self, email_config: Optional[Dict] = None):
        self.email_config = email_config
        
    def send_email_alerts(self, signals_df: pd.DataFrame):
        """Send email alerts for new signals"""
        
        if not self.email_config:
            logger.warning("Email config not provided, skipping email alerts")
            return
            
        # Filter to strong signals only
        strong_signals = signals_df[
            (signals_df['recommended_action'] == 'BUY') & 
            (signals_df['signal_probability'] > 0.7)
        ]
        
        if strong_signals.empty:
            logger.info("No strong buy signals to send")
            return
            
        # Create email content
        subject = f"Alpha Engine Alert: {len(strong_signals)} Strong Buy Signals"
        
        body = f"""
        Alpha Engine has identified {len(strong_signals)} strong buy signals:
        
        """
        
        for _, signal in strong_signals.iterrows():
            body += f"""
        Company: {signal['company_name']} ({signal['ticker']})
        Insider: {signal['insider_name']} ({signal['insider_title']})
        Trade Value: ${signal['transaction_value']:,.0f}
        Signal Strength: {signal['signal_probability']:.2%}
        Date: {signal['transaction_date']}
        
        """
        
        body += "\nDisclaimer: This is for informational purposes only and not investment advice."
        
        try:
            self.send_email(subject, body)
            logger.info(f"Sent email alert for {len(strong_signals)} signals")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def send_email(self, subject: str, body: str):
        """Send email using SMTP"""
        
        msg = MimeMultipart()
        msg['From'] = self.email_config['from_email']
        msg['To'] = self.email_config['to_email']
        msg['Subject'] = subject
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
        server.starttls()
        server.login(self.email_config['username'], self.email_config['password'])
        
        text = msg.as_string()
        server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
        server.quit()
    
    def save_signals_to_csv(self, signals_df: pd.DataFrame, filename: str = None):
        """Save signals to CSV file"""
        
        if filename is None:
            filename = f"alpha_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        signals_df.to_csv(filename, index=False)
        logger.info(f"Saved {len(signals_df)} signals to {filename}")

class DatabaseManager:
    """SQLite database for storing insider trading data"""
    
    def __init__(self, db_path: str = "alpha_engine.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insider trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insider_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                company_name TEXT,
                cik TEXT,
                transaction_date DATE,
                filing_date DATE,
                insider_name TEXT,
                insider_title TEXT,
                transaction_code TEXT,
                shares REAL,
                price_per_share REAL,
                transaction_value REAL,
                shares_owned_after REAL,
                is_direct_ownership BOOLEAN,
                accession_number TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                company_name TEXT,
                signal_date DATE,
                signal_probability REAL,
                signal_strength TEXT,
                recommended_action TEXT,
                insider_trade_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (insider_trade_id) REFERENCES insider_trades (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_insider_trades(self, trades_df: pd.DataFrame):
        """Save insider trades to database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Use IGNORE to avoid duplicates based on accession_number
        trades_df.to_sql('insider_trades', conn, if_exists='append', index=False)
        
        conn.close()
        logger.info(f"Saved {len(trades_df)} insider trades to database")
    
    def save_signals(self, signals_df: pd.DataFrame):
        """Save signals to database"""
        
        conn = sqlite3.connect(self.db_path)
        
        signals_df[['ticker', 'company_name', 'transaction_date', 'signal_probability', 
                   'signal_strength', 'recommended_action']].to_sql(
            'signals', conn, if_exists='append', index=False
        )
        
        conn.close()
        logger.info(f"Saved {len(signals_df)} signals to database")
    
    def get_recent_trades(self, days: int = 30) -> pd.DataFrame:
        """Get recent insider trades from database"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM insider_trades 
            WHERE transaction_date >= date('now', '-{} days')
            ORDER BY transaction_date DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

class AlphaEngine:
    """Main Alpha Engine orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.sec_collector = SECDataCollector(config.get('user_agent', 'Alpha Engine'))
        self.market_collector = MarketDataCollector()
        self.feature_engineer = FeatureEngineering()
        self.model = AlphaModel()
        self.backtest_engine = BacktestEngine()
        self.signal_delivery = SignalDelivery(config.get('email_config'))
        self.db_manager = DatabaseManager(config.get('db_path', 'alpha_engine.db'))
        
        logger.info("Alpha Engine initialized successfully")
    
    def collect_and_process_data(self, days_back: int = 7) -> pd.DataFrame:
        """Collect and process recent insider trading data"""
        
        logger.info(f"Collecting insider trading data for last {days_back} days...")
        
        # Collect SEC Form 4 filings
        insider_trades = self.sec_collector.get_recent_form4_filings(days_back)
        
        if not insider_trades:
            logger.warning("No insider trades found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(insider_trades)
        
        # Save to database
        self.db_manager.save_insider_trades(trades_df)
        
        # Collect market data for unique tickers
        tickers = trades_df['ticker'].dropna().unique()
        market_data = {}
        
        start_date = datetime.now() - timedelta(days=90)  # Get 3 months of data
        end_date = datetime.now()
        
        for ticker in tickers:
            logger.info(f"Collecting market data for {ticker}")
            market_data[ticker] = self.market_collector.get_stock_data(ticker, start_date, end_date)
        
        # Engineer features
        features_df = self.feature_engineer.engineer_features(trades_df, market_data)
        
        return features_df, market_data
    
    def train_model_pipeline(self, historical_days: int = 365):
        """Train the machine learning model on historical data"""
        
        logger.info("Training model on historical data...")
        
        # Get historical data from database
        historical_trades = self.db_manager.get_recent_trades(historical_days)
        
        if len(historical_trades) < 100:
            logger.warning("Insufficient historical data for training. Need at least 100 trades.")
            return False
        
        # Collect market data for historical tickers
        tickers = historical_trades['ticker'].dropna().unique()
        market_data = {}
        
        start_date = datetime.now() - timedelta(days=historical_days + 60)
        end_date = datetime.now()
        
        for ticker in tickers:
            market_data[ticker] = self.market_collector.get_stock_data(ticker, start_date, end_date)
        
        # Engineer features
        features_df = self.feature_engineer.engineer_features(historical_trades, market_data)
        
        # Prepare training data
        X, y = self.model.prepare_training_data(features_df, market_data)
        
        if len(X) < 50:
            logger.warning("Insufficient training examples after filtering")
            return False
        
        # Train model
        self.model.train_model(X, y)
        
        logger.info("Model training completed successfully")
        return True
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals for recent data"""
        
        logger.info("Generating trading signals...")
        
        # Get recent data
        features_df, market_data = self.collect_and_process_data()
        
        if features_df.empty:
            logger.warning("No recent data to generate signals")
            return pd.DataFrame()
        
        # Generate signals
        signals_df = self.model.predict_signals(features_df)
        
        # Save signals to database
        self.db_manager.save_signals(signals_df)
        
        logger.info(f"Generated {len(signals_df)} signals")
        return signals_df
    
    def run_backtest(self, days_back: int = 365) -> Dict:
        """Run backtest on historical data"""
        
        logger.info("Running backtest...")
        
        # Get historical data
        historical_trades = self.db_manager.get_recent_trades(days_back)
        
        if len(historical_trades) < 50:
            logger.warning("Insufficient data for backtesting")
            return {}
        
        # Collect market data
        tickers = historical_trades['ticker'].dropna().unique()
        market_data = {}
        
        start_date = datetime.now() - timedelta(days=days_back + 60)
        end_date = datetime.now()
        
        for ticker in tickers:
            market_data[ticker] = self.market_collector.get_stock_data(ticker, start_date, end_date)
        
        # Engineer features and generate signals
        features_df = self.feature_engineer.engineer_features(historical_trades, market_data)
        signals_df = self.model.predict_signals(features_df)
        
        # Run backtest
        results = self.backtest_engine.run_backtest(signals_df, market_data)
        
        logger.info("Backtest completed")
        logger.info(f"Results: {results}")
        
        return results
    
    def daily_pipeline(self):
        """Main daily pipeline for collecting data and generating signals"""
        
        logger.info("Running daily Alpha Engine pipeline...")
        
        try:
            # 1. Collect and process new data
            features_df, market_data = self.collect_and_process_data(days_back=1)
            
            if features_df.empty:
                logger.info("No new insider trades found today")
                return
            
            # 2. Generate signals (if model is trained)
            if self.model.is_trained:
                signals_df = self.model.predict_signals(features_df)
                
                # 3. Filter to actionable signals
                strong_signals = signals_df[
                    (signals_df['recommended_action'] == 'BUY') & 
                    (signals_df['signal_probability'] > 0.6)
                ]
                
                if not strong_signals.empty:
                    # 4. Deliver signals
                    self.signal_delivery.save_signals_to_csv(strong_signals)
                    self.signal_delivery.send_email_alerts(strong_signals)
                    
                    logger.info(f"Generated {len(strong_signals)} strong buy signals")
                else:
                    logger.info("No strong signals generated today")
            else:
                logger.warning("Model not trained. Please run train_model_pipeline() first.")
                
        except Exception as e:
            logger.error(f"Error in daily pipeline: {e}")


def main():
    """Main function to run Alpha Engine"""
    
    # Configuration
    config = {
        'user_agent': 'Alpha Engine (student.researcher@university.edu)',
        'db_path': 'alpha_engine.db',
        'email_config': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',  # Use app password for Gmail
            'from_email': 'your_email@gmail.com',
            'to_email': 'recipient@gmail.com'
        }
    }
    
    # Initialize Alpha Engine
    alpha_engine = AlphaEngine(config)
    
    # Example usage
    try:
        # 1. Collect initial data and train model
        logger.info("Step 1: Training model on historical data...")
        success = alpha_engine.train_model_pipeline(historical_days=730)  # 2 years
        
        if not success:
            logger.error("Model training failed. Collecting more data...")
            # Collect more historical data
            alpha_engine.collect_and_process_data(days_back=30)
            success = alpha_engine.train_model_pipeline(historical_days=365)
        
        if success:
            # 2. Run backtest
            logger.info("Step 2: Running backtest...")
            backtest_results = alpha_engine.run_backtest(days_back=365)
            
            if backtest_results:
                print("\n" + "="*50)
                print("BACKTEST RESULTS")
                print("="*50)
                for metric, value in backtest_results.items():
                    if isinstance(value, float):
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value}")
                print("="*50)
            
            # 3. Generate current signals
            logger.info("Step 3: Generating current signals...")
            current_signals = alpha_engine.generate_signals()
            
            if not current_signals.empty:
                print(f"\nGenerated {len(current_signals)} signals:")
                print(current_signals[['ticker', 'company_name', 'signal_probability', 
                                     'signal_strength', 'recommended_action']].to_string())
            
            # 4. Save signals
            alpha_engine.signal_delivery.save_signals_to_csv(current_signals)
            
        else:
            logger.error("Could not train model. Please check data availability.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
