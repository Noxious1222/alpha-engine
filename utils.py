import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def get_sp500_tickers() -> List[str]:
    """Get list of S&P 500 tickers"""
    try:
        # Get S&P 500 list from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_df = tables[0]
        return sp500_df['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Error getting S&P 500 tickers: {e}")
        # Fallback to major tickers
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

def calculate_returns(prices: pd.Series, periods: List[int]) -> pd.DataFrame:
    """Calculate returns for multiple periods"""
    returns_df = pd.DataFrame(index=prices.index)
    
    for period in periods:
        returns_df[f'return_{period}d'] = prices.pct_change(periods=period)
    
    return returns_df

def get_market_cap(ticker: str) -> Optional[float]:
    """Get market capitalization for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('marketCap')
    except Exception as e:
        logger.warning(f"Error getting market cap for {ticker}: {e}")
        return None

def validate_ticker(ticker: str) -> bool:
    """Validate if ticker exists and has data"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return not hist.empty
    except:
        return False

def clean_insider_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate insider trading data"""
    if df.empty:
        return df
    
    # Remove rows with missing critical data
    required_cols = ['ticker', 'transaction_date', 'transaction_code', 'shares']
    for col in required_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    # Filter to meaningful transaction codes
    meaningful_codes = ['P', 'S', 'A', 'M']  # Purchase, Sale, Award, Exercise
    if 'transaction_code' in df.columns:
        df = df[df['transaction_code'].isin(meaningful_codes)]
    
    # Remove trades with zero or negative shares
    if 'shares' in df.columns:
        df = df[df['shares'] > 0]
    
    # Remove penny stocks (price < $1)
    if 'price_per_share' in df.columns:
        df = df[df['price_per_share'] >= 1.0]
    
    return df

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if returns.std() == 0:
        return 0
    
    excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns / returns.std() * np.sqrt(252)  # Annualized

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max) - 1
    return drawdown.min()

def format_currency(amount: float) -> str:
    """Format currency with appropriate suffixes"""
    if abs(amount) >= 1e9:
        return f"${amount/1e9:.1f}B"
    elif abs(amount) >= 1e6:
        return f"${amount/1e6:.1f}M"
    elif abs(amount) >= 1e3:
        return f"${amount/1e3:.1f}K"
    else:
        return f"${amount:.0f}"

def get_sector_mapping() -> Dict[str, str]:
    """Get ticker to sector mapping"""
    # This would typically come from a data provider
    # For demo purposes, using a small mapping
    return {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'TSLA': 'Consumer Discretionary',
        'META': 'Technology',
        'NVDA': 'Technology',
        'JPM': 'Financials',
        'JNJ': 'Healthcare',
        'V': 'Financials'
    }

def rate_limit_wrapper(func, delay: float = 0.1):
    """Wrapper to add rate limiting to functions"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        time.sleep(delay)
        return result
    return wrapper

# =============================================================================
# scheduler.py - Task Scheduler
# =============================================================================

import schedule
import time
import threading
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AlphaEngineScheduler:
    """Scheduler for Alpha Engine tasks"""
    
    def __init__(self, alpha_engine):
        self.alpha_engine = alpha_engine
        self.running = False
        self.thread = None
        
    def setup_schedule(self):
        """Setup the task schedule"""
        
        # Daily data collection and signal generation (after market close)
        schedule.every().day.at("18:00").do(self.daily_pipeline_job)
        
        # Weekly model retraining (Sundays)
        schedule.every().sunday.at("10:00").do(self.weekly_retrain_job)
        
        # Database cleanup (monthly)
        schedule.every().month.do(self.cleanup_job)
        
        logger.info("Schedule setup completed")
    
    def daily_pipeline_job(self):
        """Daily pipeline job"""
        try:
            logger.info("Running daily pipeline job...")
            self.alpha_engine.daily_pipeline()
            logger.info("Daily pipeline job completed successfully")
        except Exception as e:
            logger.error(f"Error in daily pipeline job: {e}")
    
    def weekly_retrain_job(self):
        """Weekly model retraining job"""
        try:
            logger.info("Running weekly model retraining...")
            success = self.alpha_engine.train_model_pipeline(historical_days=730)
            if success:
                logger.info("Weekly retraining completed successfully")
            else:
                logger.warning("Weekly retraining failed")
        except Exception as e:
            logger.error(f"Error in weekly retraining: {e}")
    
    def cleanup_job(self):
        """Monthly cleanup job"""
        try:
            logger.info("Running monthly cleanup...")
            # Add cleanup logic here (e.g., archive old data, remove logs)
            logger.info("Monthly cleanup completed")
        except Exception as e:
            logger.error(f"Error in cleanup job: {e}")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.setup_schedule()
        self.running = True
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.thread = threading.Thread(target=run_scheduler, daemon=True)
        self.thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Scheduler stopped")

# =============================================================================
# config_manager.py - Configuration Management
# =============================================================================

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for Alpha Engine"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables"""
        
        # Default configuration
        default_config = {
            'sec': {
                'user_agent': 'Alpha Engine (research@example.com)',
                'rate_limit': 10,
                'retry_attempts': 3,
                'timeout': 30
            },
            'database': {
                'path': 'alpha_engine.db',
                'backup_interval': 24
            },
            'model': {
                'algorithm': 'xgboost',
                'validation_split': 0.2,
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'signals': {
                'min_probability': 0.6,
                'holding_period_days': 10,
                'max_signals_per_day': 20,
                'min_trade_value': 10000
            },
            'notifications': {
                'email': {'enabled': False},
                'csv_export': {'enabled': True, 'directory': './signals'},
                'slack': {'enabled': False}
            },
            'logging': {
                'level': 'INFO',
                'file': 'alpha_engine.log'
            }
        }
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        # Override with environment variables
        env_overrides = {
            'sec.user_agent': os.getenv('SEC_USER_AGENT'),
            'database.path': os.getenv('DATABASE_PATH'),
            'notifications.email.username': os.getenv('EMAIL_USERNAME'),
            'notifications.email.password': os.getenv('EMAIL_PASSWORD'),
            'notifications.email.from_email': os.getenv('EMAIL_FROM'),
            'notifications.email.to_email': os.getenv('EMAIL_TO'),
            'notifications.slack.webhook_url': os.getenv('SLACK_WEBHOOK_URL')
        }
        
        for key, value in env_overrides.items():
            if value is not None:
                self._set_nested_config(default_config, key, value)
        
        return default_config
    
    def _set_nested_config(self, config: Dict, key: str, value: Any):
        """Set nested configuration value"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        current = self.config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except KeyError:
            return default
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

# =============================================================================
# cli.py - Command Line Interface
# =============================================================================

import click
import sys
from alpha_engine import AlphaEngine
from config_manager import ConfigManager
from scheduler import AlphaEngineScheduler
import logging

@click.group()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Alpha Engine CLI - AI-powered insider trading signal generator"""
    
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config_manager = ConfigManager(config)
    
    # Initialize Alpha Engine
    alpha_engine = AlphaEngine(config_manager.config)
    
    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['alpha_engine'] = alpha_engine
    ctx.obj['config_manager'] = config_manager

@cli.command()
@click.option('--days', default=730, help='Number of historical days for training')
@click.pass_context
def train(ctx, days):
    """Train the machine learning model"""
    alpha_engine = ctx.obj['alpha_engine']
    
    click.echo(f"Training model on {days} days of historical data...")
    
    success = alpha_engine.train_model_pipeline(historical_days=days)
    
    if success:
        click.echo("✅ Model training completed successfully!")
    else:
        click.echo("❌ Model training failed. Check logs for details.")
        sys.exit(1)

@cli.command()
@click.option('--days', default=7, help='Number of days to look back for new data')
@click.pass_context
def collect(ctx, days):
    """Collect insider trading data"""
    alpha_engine = ctx.obj['alpha_engine']
    
    click.echo(f"Collecting insider trading data for last {days} days...")
    
    features_df, market_data = alpha_engine.collect_and_process_data(days_back=days)
    
    if not features_df.empty:
        click.echo(f"✅ Collected {len(features_df)} insider trades")
    else:
        click.echo("⚠️  No new insider trades found")

@cli.command()
@click.pass_context
def signals(ctx):
    """Generate trading signals"""
    alpha_engine = ctx.obj['alpha_engine']
    
    click.echo("Generating trading signals...")
    
    signals_df = alpha_engine.generate_signals()
    
    if not signals_df.empty:
        buy_signals = len(signals_df[signals_df['recommended_action'] == 'BUY'])
        click.echo(f"✅ Generated {len(signals_df)} signals ({buy_signals} BUY signals)")
        
        # Display top signals
        strong_signals = signals_df[
            (signals_df['recommended_action'] == 'BUY') &
            (signals_df['signal_probability'] > 0.7)
        ].head(5)
        
        if not strong_signals.empty:
            click.echo("\n🎯 Top Strong Buy Signals:")
            for _, signal in strong_signals.iterrows():
                click.echo(f"  • {signal['ticker']} - {signal['company_name']}")
                click.echo(f"    Probability: {signal['signal_probability']:.2%}")
                click.echo(f"    Strength: {signal['signal_strength']}")
                click.echo("")
    else:
        click.echo("⚠️  No signals generated")

@cli.command()
@click.option('--days', default=365, help='Number of days for backtesting')
@click.pass_context
def backtest(ctx, days):
    """Run strategy backtest"""
    alpha_engine = ctx.obj['alpha_engine']
    
    click.echo(f"Running backtest on {days} days of data...")
    
    results = alpha_engine.run_backtest(days_back=days)
    
    if results:
        click.echo("📊 Backtest Results:")
        click.echo("=" * 40)
        for metric, value in results.items():
            if isinstance(value, float):
                click.echo(f"{metric}: {value:.4f}")
            else:
                click.echo(f"{metric}: {value}")
        click.echo("=" * 40)
    else:
        click.echo("❌ Backtest failed. Check logs for details.")

@cli.command()
@click.pass_context
def run(ctx):
    """Run daily pipeline"""
    alpha_engine = ctx.obj['alpha_engine']
    
    click.echo("Running daily Alpha Engine pipeline...")
    alpha_engine.daily_pipeline()
    click.echo("✅ Daily pipeline completed")

@cli.command()
@click.pass_context
def schedule(ctx):
    """Start scheduled execution"""
    alpha_engine = ctx.obj['alpha_engine']
    
    click.echo("Starting Alpha Engine scheduler...")
    
    scheduler = AlphaEngineScheduler(alpha_engine)
    scheduler.start()
    
    try:
        click.echo("⏰ Scheduler is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n🛑 Stopping scheduler...")
        scheduler.stop()
        click.echo("✅ Scheduler stopped")

@cli.command()
@click.option('--port', default=8501, help='Port for dashboard')
def dashboard(port):
    """Launch web dashboard"""
    import subprocess
    import sys
    
    click.echo(f"Starting dashboard on port {port}...")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py', '--server.port', str(port)])
    except KeyboardInterrupt:
        click.echo("\n🛑 Dashboard stopped")

if __name__ == '__main__':
    cli()

# =============================================================================
# tests/test_alpha_engine.py - Basic Tests
# =============================================================================

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

# Import your modules (adjust paths as needed)
from alpha_engine import FeatureEngineering, AlphaModel, BacktestEngine

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        self.fe = FeatureEngineering()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'AAPL'],
            'transaction_date': [
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=1),
                datetime.now()
            ],
            'transaction_code': ['P', 'S', 'P'],
            'shares': [1000, 500, 2000],
            'transaction_value': [50000, 25000, 100000],
            'insider_title': ['CEO', 'CFO', 'Director']
        })
    
    def test_basic_features(self):
        """Test basic feature engineering"""
        result = self.fe.engineer_features(self.sample_data, {})
        
        # Check if basic features are created
        self.assertIn('is_purchase', result.columns)
        self.assertIn('is_sale', result.columns)
        self.assertIn('log_transaction_value', result.columns)
        
        # Check values
        self.assertTrue(result['is_purchase'].iloc[0])
        self.assertFalse(result['is_purchase'].iloc[1])
    
    def test_cluster_features(self):
        """Test cluster feature calculation"""
        result = self.fe.engineer_features(self.sample_data, {})
        
        # AAPL has 2 trades, so cluster count should be 1 for each AAPL trade
        aapl_trades = result[result['ticker'] == 'AAPL']
        self.assertTrue(all(aapl_trades['insider_cluster_5d'] >= 1))

class TestAlphaModel(unittest.TestCase):
    
    def setUp(self):
        self.model = AlphaModel()
        
        # Create sample features
        np.random.seed(42)
        self.sample_features = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'transaction_date': [datetime.now() - timedelta(days=i) for i in range(100)],
            'is_purchase': np.random.choice([True, False], 100),
            'log_transaction_value': np.random.normal(10, 2, 100),
            'is_ceo': np.random.choice([True, False], 100),
            'stock_return_5d': np.random.normal(0, 0.1, 100),
            'volume_ratio': np.random.exponential(0.1, 100)
        })
    
    def test_model_training(self):
        """Test model training"""
        # Create mock market data
        dates = pd.date_range(start=datetime.now() - timedelta(days=120), 
                             end=datetime.now(), freq='D')
        market_data = {
            'AAPL': pd.DataFrame({
                'Close': np.random.normal(150, 10, len(dates))
            }, index=dates)
        }
        
        X, y = self.model.prepare_training_data(self.sample_features, market_data)
        
        if len(X) > 10:  # Only test if we have enough data
            self.model.train_model(X, y)
            self.assertTrue(self.model.is_trained)

class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        self.backtest = BacktestEngine(initial_capital=100000)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Create sample trades
        trades_df = pd.DataFrame({
            'return': [0.05, -0.02, 0.08, -0.01, 0.03]
        })
        
        metrics = self.backtest.calculate_performance_metrics(trades_df)
        
        # Check if all expected metrics are present
        expected_metrics = ['total_trades', 'win_rate', 'avg_return_per_trade', 
                           'total_return', 'volatility', 'sharpe_ratio']
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check win rate calculation
        self.assertEqual(metrics['win_rate'], 0.6)  # 3 out of 5 positive returns

if __name__ == '__main__':
    unittest.main()
