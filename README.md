# Alpha Engine 🚀

An AI-powered insider trading signal generator that analyzes SEC Form 4 filings to identify profitable trading opportunities.

## Features

- **Automated Data Collection**: Fetches SEC Form 4 filings using the `edgartools` library
- **Machine Learning Signals**: XGBoost model trained on historical insider trading patterns
- **Feature Engineering**: Advanced features including cluster analysis, market context, and insider roles
- **Backtesting Framework**: Comprehensive backtesting with performance metrics
- **Signal Delivery**: Email alerts and CSV exports for actionable signals
- **Cost-Efficient**: Designed for solo developers using free/low-cost data sources

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/alpha-engine.git
cd alpha-engine
pip install -r requirements.txt
Basic Usage
pythonfrom alpha_engine import AlphaEngine

# Configure the engine
config = {
    'user_agent': 'Alpha Engine (your.email@example.com)',
    'db_path': 'alpha_engine.db',
    'email_config': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_app_password',
        'from_email': 'your_email@gmail.com',
        'to_email': 'recipient@gmail.com'
    }
}

# Initialize and run
alpha_engine = AlphaEngine(config)

# Train model on historical data
alpha_engine.train_model_pipeline(historical_days=730)

# Generate current signals
signals = alpha_engine.generate_signals()
print(signals)

# Run backtest
results = alpha_engine.run_backtest()
print(results)
Command Line Usage
bash# Run daily pipeline
python -m alpha_engine.main

# Train model only
python -c "from alpha_engine import AlphaEngine; AlphaEngine(config).train_model_pipeline()"
Architecture
Data Sources

SEC EDGAR: Form 4 filings via edgartools
Yahoo Finance: Historical price and volume data via yfinance
Market Data: SPY benchmark and market indicators

Feature Engineering

Transaction characteristics (size, type, insider role)
Market context (prior returns, volume ratios)
Cluster analysis (multiple insiders trading)
Technical indicators (volatility, moving averages)

Model

Algorithm: XGBoost Classifier
Target: 10-day excess returns > 2%
Validation: Time series split to prevent lookahead bias
Features: 20+ engineered features from insider and market data

Signal Generation

Buy Signals: High-probability insider purchases
Confidence Levels: Weak, Medium, Strong, Very Strong
Filtering: Minimum probability thresholds and position sizing

Performance Metrics
The backtesting framework provides comprehensive performance analysis:

Returns: Total return, annualized return (CAGR)
Risk: Volatility, Sharpe ratio, maximum drawdown
Trade Analysis: Win rate, average winner/loser, total trades
Benchmark: Comparison vs market (SPY) performance

Configuration
Email Alerts Setup
For Gmail, you'll need to:

Enable 2-factor authentication
Generate an app-specific password
Use the app password in the config

pythonemail_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_16_char_app_password',  # Not your regular password!
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient@gmail.com'
}
SEC Compliance
Make sure to set a proper user agent string with your contact information:
pythonuser_agent = 'Alpha Engine (your.name@university.edu)'
The SEC requires this for API access and may block requests without proper identification.
Data Storage
The system uses SQLite for local data storage with two main tables:

insider_trades: Raw insider trading data from Form 4 filings
signals: Generated trading signals with probabilities and recommendations

Deployment Options
Local Development

Run scripts locally with cron jobs for scheduling
SQLite database for data storage
CSV file outputs for signal delivery

Cloud Deployment (Free Tier)

AWS Lambda + EventBridge for scheduled execution
Google Cloud Functions + Cloud Scheduler
Render/Railway for web dashboard hosting

Scaling Considerations

Start with major stocks (S&P 500) to limit data volume
Implement rate limiting for SEC API calls
Consider PostgreSQL for larger datasets

Research Background
This implementation is based on extensive academic research showing that insider trading is predictive of future stock returns. Key findings:

Insider purchases show stronger predictive power than sales
CEO and director trades are more informative than other insiders
Cluster buying (multiple insiders) increases signal strength
Optimal holding periods are typically 5-20 trading days

Legal Disclaimer
This tool is for educational and informational purposes only. It does not constitute investment advice.

Past performance does not guarantee future results
All investments carry risk of loss
Consult with a licensed financial advisor before making investment decisions
The authors are not responsible for any financial losses incurred

Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.
Development Setup
bashgit clone https://github.com/yourusername/alpha-engine.git
cd alpha-engine
pip install -e .
pip install -r requirements-dev.txt
Running Tests
bashpytest tests/
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

edgartools for SEC data access
Academic research on insider trading by Nejat Seyhun, H. Nejat Seyhun, and others
Project Layline for open insider trading datasets

Roadmap
Version 0.2.0

 Web dashboard using Streamlit
 Real-time signal notifications via Slack/Discord
 Support for options trading analysis
 Enhanced risk management features

Version 0.3.0

 Multi-factor model incorporating earnings, news sentiment
 Portfolio optimization and position sizing
 Paper trading integration
 Performance attribution analysis

Support
For questions or support:

Open an issue on GitHub
Join our Discord community
Email: support@alpha-engine.dev


⚠️ Important: This is a research project. Always do your own due diligence before making any investment decisions.
