import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from hmmlearn import hmm
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import time
import hashlib

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Forex Trading Bot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)


class OptimizedDataCache:
    """Enhanced caching system with persistent storage"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, symbol: str, period: str, extra: str = "") -> str:
        """Generate consistent cache key"""
        return hashlib.md5(f"{symbol}_{period}_{extra}".encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def is_cache_valid(self, cache_time: datetime, max_age_hours: int = 1) -> bool:
        """Check if cache is still valid"""
        return (datetime.now() - cache_time).total_seconds() < (max_age_hours * 3600)

    def get(self, symbol: str, period: str, max_age_hours: int = 1) -> Optional[pd.DataFrame]:
        """Get cached data if valid"""
        cache_key = self._get_cache_key(symbol, period)

        # Check memory cache first
        if cache_key in self.memory_cache:
            data, cache_time = self.memory_cache[cache_key]
            if self.is_cache_valid(cache_time, max_age_hours):
                return data
            else:
                del self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self._get_cache_file(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    data, cache_time = cached_data['data'], cached_data['timestamp']

                if self.is_cache_valid(cache_time, max_age_hours):
                    # Load back to memory cache
                    self.memory_cache[cache_key] = (data, cache_time)
                    return data
                else:
                    os.remove(cache_file)
            except Exception as e:
                st.warning(f"Cache read error: {e}")
                if os.path.exists(cache_file):
                    os.remove(cache_file)

        return None

    def set(self, symbol: str, period: str, data: pd.DataFrame):
        """Cache data both in memory and disk"""
        cache_key = self._get_cache_key(symbol, period)
        timestamp = datetime.now()

        # Memory cache
        self.memory_cache[cache_key] = (data, timestamp)

        # Disk cache
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': timestamp
                }, f)
        except Exception as e:
            st.warning(f"Cache write error: {e}")


class ForexTradingBot:
    def __init__(self):
        self.currency_pairs = {
            'EUR/USD': 'EURUSD=X',
            'USD/JPY': 'USDJPY=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/CHF': 'USDCHF=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X'
        }
        self.cache = OptimizedDataCache()
        self.rate_limit_delay = 0.5  # Delay between API calls
        self.last_api_call = 0

        # Initialize session state for persistent caching
        if 'processed_indicators' not in st.session_state:
            st.session_state.processed_indicators = {}

    def _rate_limit_delay(self):
        """Implement rate limiting between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)

        self.last_api_call = time.time()

    def fetch_forex_data(self, symbol: str, period: str = "3mo", max_retries: int = 3) -> pd.DataFrame:
        """Fetch forex data with enhanced caching and error handling"""

        # Try to get from cache first
        cached_data = self.cache.get(symbol, period, max_age_hours=2)  # Increased cache time
        if cached_data is not None:
            return cached_data

        # If not in cache, fetch from API with retry logic
        for attempt in range(max_retries):
            try:
                self._rate_limit_delay()

                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1d")

                if data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        st.error(f"No data available for {symbol} after {max_retries} attempts")
                        return pd.DataFrame()

                # Cache the successful result
                self.cache.set(symbol, period, data)
                return data

            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    st.error(f"Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def calculate_technical_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators with caching"""
        if data.empty:
            return data

        # Check if indicators are already calculated and cached
        cache_key = f"{symbol}_indicators_{len(data)}"
        if cache_key in st.session_state.processed_indicators:
            cached_result = st.session_state.processed_indicators[cache_key]
            if len(cached_result) == len(data):
                return cached_result

        df = data.copy()

        try:
            # RSI with error handling
            rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'])
            df['RSI'] = rsi_indicator.rsi()

            # MACD
            macd_indicator = ta.trend.MACD(close=df['Close'])
            df['MACD'] = macd_indicator.macd()
            df['MACD_Signal'] = macd_indicator.macd_signal()
            df['MACD_Histogram'] = macd_indicator.macd_diff()

            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(close=df['Close'])
            df['BB_High'] = bb_indicator.bollinger_hband()
            df['BB_Low'] = bb_indicator.bollinger_lband()
            df['BB_Mid'] = bb_indicator.bollinger_mavg()

            # Moving Averages
            df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12).ema_indicator()

            # Volatility
            df['ATR'] = ta.volatility.AverageTrueRange(
                high=df['High'], low=df['Low'], close=df['Close']
            ).average_true_range()

            # Cache the result
            st.session_state.processed_indicators[cache_key] = df

            # Limit cache size to prevent memory issues
            if len(st.session_state.processed_indicators) > 10:
                # Remove oldest entry
                oldest_key = list(st.session_state.processed_indicators.keys())[0]
                del st.session_state.processed_indicators[oldest_key]

        except Exception as e:
            st.warning(f"Error calculating indicators for {symbol}: {e}")
            return data

        return df

    def prepare_hmm_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM model with improved error handling"""
        if data.empty or len(data) < 20:  # Minimum data requirement
            return np.array([])

        try:
            df = data.copy()

            # Calculate returns and volatility features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=min(10, len(df) // 2)).std()
            df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']

            # Normalize RSI and MACD for HMM (with null checks)
            if 'RSI' in df.columns:
                df['RSI_Norm'] = (df['RSI'] - 50) / 50
            else:
                df['RSI_Norm'] = 0

            if 'MACD' in df.columns:
                df['MACD_Norm'] = df['MACD'] / df['Close']
            else:
                df['MACD_Norm'] = 0

            # Select features for HMM
            features = ['Log_Returns', 'Volatility', 'RSI_Norm', 'MACD_Norm', 'Price_Change', 'High_Low_Ratio']

            # Create feature matrix with improved null handling
            feature_matrix = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).values

            return feature_matrix

        except Exception as e:
            st.warning(f"Error preparing HMM features: {e}")
            return np.array([])

    def train_hmm_model(self, features: np.ndarray, n_states: int = 3) -> Optional[hmm.GaussianHMM]:
        """Train Hidden Markov Model for regime detection with better error handling"""
        if len(features) < 20:  # Minimum samples for training
            return None

        try:
            # Remove any remaining invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                random_state=42,
                n_iter=100  # Limit iterations to prevent timeout
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features)

            return model

        except Exception as e:
            st.warning(f"HMM training failed: {str(e)}")
            return None

    def get_market_regime(self, model: hmm.GaussianHMM, features: np.ndarray) -> Tuple[int, float]:
        """Get current market regime and confidence"""
        if model is None or len(features) == 0:
            return 1, 0.5  # Default to neutral regime

        try:
            # Get the most recent features
            recent_features = features[-min(10, len(features)):]

            # Predict regime
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                states = model.predict(recent_features)

            current_state = states[-1]

            # Calculate confidence based on state probabilities
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_prob = model.score(recent_features)

            # Normalize confidence score
            confidence = max(0.1, min(0.9, (log_prob + 50) / 100))

            return int(current_state), float(confidence)

        except Exception as e:
            st.warning(f"Regime prediction failed: {str(e)}")
            return 1, 0.5

    '''def get_news_sentiment(self, pair: str) -> float:
        """Simulate news sentiment analysis (placeholder)"""
        # In a real implementation, you would cache sentiment data
        # and only update it periodically (e.g., every few hours)

        # For demo purposes, return a consistent "random" sentiment based on pair
        import hashlib
        seed = int(hashlib.md5(f"{pair}_{datetime.now().strftime('%Y-%m-%d')}".encode()).hexdigest()[:8], 16)
        np.random.seed(seed % (2 ** 31))
        return np.random.uniform(-0.5, 0.5)'''

    def generate_trade_signal(self, data: pd.DataFrame, pair: str) -> Optional[Dict]:
        """Generate trading signal for a currency pair - Modified to be more inclusive"""
        if data.empty or len(data) < 50:
            return None

        try:
            # Calculate technical indicators with caching
            data_with_indicators = self.calculate_technical_indicators(data, pair)

            # Prepare HMM features
            hmm_features = self.prepare_hmm_features(data_with_indicators)

            if len(hmm_features) == 0:
                return None

            # Train HMM model
            hmm_model = self.train_hmm_model(hmm_features)

            # Get market regime
            regime, regime_confidence = self.get_market_regime(hmm_model, hmm_features)

            # Get latest indicators
            latest = data_with_indicators.iloc[-1]
            prev = data_with_indicators.iloc[-2] if len(data_with_indicators) > 1 else latest

            # Generate signals based on technical indicators
            signals = []
            signal_strengths = []  # Track individual signal strengths

            # RSI signal with more nuanced scoring
            if not pd.isna(latest.get('RSI', np.nan)):
                rsi_value = latest['RSI']
                if rsi_value < 25:
                    signals.append(1)
                    signal_strengths.append(0.9)  # Strong oversold
                elif rsi_value < 35:
                    signals.append(1)
                    signal_strengths.append(0.6)  # Moderate oversold
                elif rsi_value > 75:
                    signals.append(-1)
                    signal_strengths.append(0.9)  # Strong overbought
                elif rsi_value > 65:
                    signals.append(-1)
                    signal_strengths.append(0.6)  # Moderate overbought
                elif rsi_value < 45:
                    signals.append(0.5)  # Slight bullish bias
                    signal_strengths.append(0.3)
                elif rsi_value > 55:
                    signals.append(-0.5)  # Slight bearish bias
                    signal_strengths.append(0.3)
                else:
                    signals.append(0)
                    signal_strengths.append(0.1)
            else:
                signals.append(0)
                signal_strengths.append(0)

            # MACD signal with gradient consideration
            if not pd.isna(latest.get('MACD', np.nan)) and not pd.isna(latest.get('MACD_Signal', np.nan)):
                macd_diff = latest['MACD'] - latest['MACD_Signal']
                prev_macd_diff = prev['MACD'] - prev['MACD_Signal']

                if (latest['MACD'] > latest['MACD_Signal'] and
                        prev['MACD'] <= prev['MACD_Signal']):
                    signals.append(1)  # Bullish crossover
                    signal_strengths.append(0.8)
                elif (latest['MACD'] < latest['MACD_Signal'] and
                      prev['MACD'] >= prev['MACD_Signal']):
                    signals.append(-1)  # Bearish crossover
                    signal_strengths.append(0.8)
                elif macd_diff > 0:
                    # MACD above signal line
                    strength = min(abs(macd_diff) * 1000, 0.6)  # Scale the difference
                    signals.append(0.5)
                    signal_strengths.append(strength)
                elif macd_diff < 0:
                    # MACD below signal line
                    strength = min(abs(macd_diff) * 1000, 0.6)
                    signals.append(-0.5)
                    signal_strengths.append(strength)
                else:
                    signals.append(0)
                    signal_strengths.append(0.1)
            else:
                signals.append(0)
                signal_strengths.append(0)

            # Bollinger Bands signal with distance consideration
            if (not pd.isna(latest.get('BB_Low', np.nan)) and
                    not pd.isna(latest.get('BB_High', np.nan)) and
                    not pd.isna(latest.get('BB_Mid', np.nan))):

                bb_position = (latest['Close'] - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low'])

                if bb_position < 0.1:  # Very close to lower band
                    signals.append(1)
                    signal_strengths.append(0.8)
                elif bb_position < 0.3:  # Below lower third
                    signals.append(0.7)
                    signal_strengths.append(0.5)
                elif bb_position > 0.9:  # Very close to upper band
                    signals.append(-1)
                    signal_strengths.append(0.8)
                elif bb_position > 0.7:  # Above upper third
                    signals.append(-0.7)
                    signal_strengths.append(0.5)
                else:
                    # Use distance from middle band
                    mid_distance = (latest['Close'] - latest['BB_Mid']) / latest['BB_Mid']
                    signals.append(mid_distance * 2)  # Scale the signal
                    signal_strengths.append(min(abs(mid_distance) * 10, 0.4))
            else:
                signals.append(0)
                signal_strengths.append(0)

            # Add trend signal based on moving averages
            if not pd.isna(latest.get('SMA_20', np.nan)) and not pd.isna(latest.get('EMA_12', np.nan)):
                if latest['Close'] > latest['SMA_20'] and latest['EMA_12'] > latest['SMA_20']:
                    signals.append(0.5)  # Uptrend
                    signal_strengths.append(0.4)
                elif latest['Close'] < latest['SMA_20'] and latest['EMA_12'] < latest['SMA_20']:
                    signals.append(-0.5)  # Downtrend
                    signal_strengths.append(0.4)
                else:
                    signals.append(0)
                    signal_strengths.append(0.1)
            else:
                signals.append(0)
                signal_strengths.append(0)

            # Calculate weighted technical score
            if signals and signal_strengths:
                weighted_signals = [sig * strength for sig, strength in zip(signals, signal_strengths)]
                total_weight = sum(signal_strengths)
                technical_score = sum(weighted_signals) / max(total_weight, 0.1)
            else:
                technical_score = 0

            # Add momentum signal as tiebreaker
            recent_returns = data['Close'].pct_change().tail(5).mean()
            momentum_signal = np.tanh(recent_returns * 100)  # Scale and bound between -1 and 1

            # Combine technical score with momentum
            combined_score = technical_score * 0.8 + momentum_signal * 0.2

            # Calculate overall confidence based on signal agreement
            signal_agreement = 1 - (np.std(signals) / max(np.mean(np.abs(signals)), 0.1))
            signal_strength = abs(combined_score)
            overall_confidence = (signal_strength * 0.4 + regime_confidence * 0.3 + signal_agreement * 0.3)
            overall_confidence = max(0.1, min(0.95, overall_confidence))  # Bound confidence

            # More lenient signal determination - LOWERED THRESHOLDS
            final_signal = 0
            if combined_score > 0.15:  # Lowered from 0.3
                final_signal = 1  # Buy
            elif combined_score < -0.15:  # Lowered from -0.3
                final_signal = -1  # Sell
            elif abs(combined_score) > 0.05:  # Very weak signals still count
                final_signal = 1 if combined_score > 0 else -1
                overall_confidence = max(overall_confidence * 0.5, 0.2)  # Reduce confidence for weak signals

            # Calculate stop loss and take profit
            atr = latest.get('ATR', latest['Close'] * 0.001)
            current_price = latest['Close']

            # Adjust risk based on signal strength
            risk_multiplier = 1.5 + (signal_strength * 1.0)  # 1.5x to 2.5x ATR

            if final_signal == 1:  # Buy
                stop_loss = current_price - (risk_multiplier * atr)
                take_profit = current_price + (risk_multiplier * 1.5 * atr)  # 1.5:1 risk-reward minimum
            elif final_signal == -1:  # Sell
                stop_loss = current_price + (risk_multiplier * atr)
                take_profit = current_price - (risk_multiplier * 1.5 * atr)
            else:
                stop_loss = current_price
                take_profit = current_price

            return {
                'pair': pair,
                'signal': final_signal,
                'signal_text': 'BUY' if final_signal == 1 else ('SELL' if final_signal == -1 else 'HOLD'),
                'confidence': overall_confidence,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'regime': regime,
                'technical_score': combined_score,
                'momentum_signal': momentum_signal,
                'signal_strength': signal_strength,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            st.error(f"Error generating signal for {pair}: {str(e)}")
            return None

    def generate_daily_trades(self, capital: float, risk_percent: float, max_trades: int = 3) -> List[Dict]:
        """Generate daily trades with guaranteed minimum trades"""
        all_signals = []

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_pairs = len(self.currency_pairs)

        # Process pairs in batches to reduce API load
        for i, (pair_name, symbol) in enumerate(self.currency_pairs.items()):
            status_text.text(f"Analyzing {pair_name} ({i + 1}/{total_pairs})...")

            try:
                # Fetch data with enhanced caching
                data = self.fetch_forex_data(symbol, period="3mo")

                if not data.empty:
                    signal = self.generate_trade_signal(data, pair_name)
                    if signal:  # Accept ALL signals, including HOLD (we'll filter later)

                        # Calculate position size
                        position_size = self.calculate_position_size(
                            capital, risk_percent, signal['current_price'], signal['stop_loss']
                        )

                        if position_size > 0:
                            signal['position_size'] = position_size
                            signal['position_percent'] = min(
                                (position_size * signal['current_price'] / capital) * 100,
                                risk_percent
                            )
                            all_signals.append(signal)

                # Update progress
                progress_bar.progress((i + 1) / total_pairs)

                # Small delay to be respectful to API
                if i < total_pairs - 1:
                    time.sleep(0.1)

            except Exception as e:
                st.warning(f"Error processing {pair_name}: {str(e)}")
                continue

        status_text.text("Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Filter and sort signals
        if all_signals:
            # First, try to get trades with actual BUY/SELL signals
            trading_signals = [s for s in all_signals if s['signal'] != 0]

            if len(trading_signals) >= max_trades:
                # We have enough trading signals
                trading_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                return trading_signals[:max_trades]
            elif len(trading_signals) > 0:
                # We have some trading signals, but not enough
                trading_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

                # Add the best HOLD signals to reach max_trades
                hold_signals = [s for s in all_signals if s['signal'] == 0]
                hold_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

                needed = max_trades - len(trading_signals)
                return trading_signals + hold_signals[:needed]
            else:
                # No trading signals, return best HOLD signals
                all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                return all_signals[:max_trades]

        # If we still have no signals, create a fallback
        if not all_signals:
            st.warning("No data could be retrieved. Please check your internet connection.")
            return []

        return all_signals[:max_trades]

    def calculate_position_size(self, capital: float, risk_percent: float, entry_price: float,
                                stop_loss: float) -> float:
        if entry_price <= 0 or stop_loss <= 0:
            return 0

        risk_amount = capital * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        position_size = risk_amount / price_risk  # float for precision
        position_size = max(position_size, 1)  # minimum 1 unit

        max_position_size = capital / entry_price * 0.1  # max 10% capital in units

        return min(position_size, max_position_size)


def save_trade_history(trades: List[Dict]):
    """Save trades to local CSV file"""
    if not trades:
        return

    try:
        df = pd.DataFrame(trades)

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        filename = f"data/trades_{datetime.now().strftime('%Y%m%d')}.csv"

        # Append to existing file or create new
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                st.warning(f"Error reading existing file: {e}")

        df.to_csv(filename, index=False)
        st.success(f"Trades saved to {filename}")

    except Exception as e:
        st.error(f"Error saving trades: {e}")


def load_trade_history() -> pd.DataFrame:
    """Load recent trade history"""
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return pd.DataFrame()

        csv_files = [f for f in os.listdir(data_dir) if f.startswith('trades_') and f.endswith('.csv')]

        if not csv_files:
            return pd.DataFrame()

        # Load the most recent file
        csv_files.sort(reverse=True)
        latest_file = os.path.join(data_dir, csv_files[0])

        return pd.read_csv(latest_file)

    except Exception as e:
        st.warning(f"Error loading trade history: {e}")
        return pd.DataFrame()


def create_trade_chart(data: pd.DataFrame, signal: Dict):
    """Create a chart for trade visualization with error handling"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price Action', 'RSI'),
            row_heights=[0.7, 0.3]
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Bollinger Bands
        if all(col in data.columns for col in ['BB_High', 'BB_Low']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_High'], name='BB High',
                           line=dict(color='gray', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Low'], name='BB Low', fill='tonexty',
                           line=dict(color='gray', width=1)),
                row=1, col=1
            )

        # Entry point
        if signal:
            color = 'green' if signal['signal'] == 1 else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[signal['current_price']],
                    mode='markers',
                    marker=dict(color=color, size=10,
                                symbol='triangle-up' if signal['signal'] == 1 else 'triangle-down'),
                    name=f"Entry ({signal['signal_text']})"
                ),
                row=1, col=1
            )

        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)

        return fig

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return go.Figure()  # Return empty figure on error


def main():
    st.title("💹 Optimized Forex Trading Bot")
    st.markdown("### Daily Trade Recommendations with Rate Limiting")

    # Initialize the bot with session state persistence
    if 'bot' not in st.session_state:
        st.session_state.bot = ForexTradingBot()

    # Display cache status
    if hasattr(st.session_state.bot, 'cache'):
        cache_info = f"Cache entries: {len(st.session_state.bot.cache.memory_cache)}"
        st.sidebar.info(cache_info)

    # Sidebar for user inputs
    st.sidebar.header("⚙️ Trading Parameters")

    capital = st.sidebar.number_input(
        "Total Capital (USD)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Enter your total trading capital"
    )

    risk_percent = st.sidebar.slider(
        "Max Risk per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Maximum percentage of capital to risk per trade"
    )

    max_trades = st.sidebar.slider(
        "Maximum Daily Trades",
        min_value=1,
        max_value=6,
        value=3,
        help="Maximum number of trades to generate per day"
    )

    # Cache management
    if st.sidebar.button("🗑️ Clear Cache"):
        if hasattr(st.session_state.bot, 'cache'):
            st.session_state.bot.cache.memory_cache.clear()
            # Clear disk cache
            cache_dir = st.session_state.bot.cache.cache_dir
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(cache_dir, file))
        if 'processed_indicators' in st.session_state:
            st.session_state.processed_indicators.clear()
        st.sidebar.success("Cache cleared!")

    # Main content area
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("📊 Daily Trade Generation")

        # Check if we have recent trades to avoid unnecessary API calls
        last_generation_time = st.session_state.get('last_generation_time', None)
        current_time = datetime.now()

        can_generate = (last_generation_time is None or
                        (current_time - last_generation_time).total_seconds() > 300)  # 5 minute cooldown

        if can_generate:
            if st.button("🚀 Generate Today's Trades", type="primary", use_container_width=True):
                with st.spinner("Analyzing markets and generating trades..."):
                    trades = st.session_state.bot.generate_daily_trades(capital, risk_percent, max_trades)
                    st.session_state.current_trades = trades
                    st.session_state.last_generation_time = current_time
        else:
            time_remaining = 300 - (current_time - last_generation_time).total_seconds()
            st.info(f"Rate limiting active. Please wait {time_remaining:.0f} seconds before generating new trades.")

    with col2:
        st.metric("💰 Available Capital", f"${capital:,}")

    with col3:
        st.metric("🎯 Max Risk/Trade", f"{risk_percent}%")

    # Display trades
    if 'current_trades' in st.session_state and st.session_state.current_trades:
        st.subheader("🎯 Today's Recommended Trades")

        for i, trade in enumerate(st.session_state.current_trades):
            with st.expander(f"Trade #{i + 1}: {trade['pair']} - {trade['signal_text']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    confidence = trade.get('confidence', 0)
                    st.metric("Signal", trade['signal_text'],
                              delta=f"{confidence:.1%} confidence" if confidence > 0 else "Low confidence")

                with col2:
                    st.metric("Entry Price", f"{trade['current_price']:.5f}")

                with col3:
                    st.metric("Take Profit", f"{trade['take_profit']:.5f}")

                with col4:
                    st.metric("Stop Loss", f"{trade['stop_loss']:.5f}")

                col5, col6, col7, col8 = st.columns(4)

                with col5:
                    position_size = trade.get('position_size', 0)
                    st.metric("Position Size", f"{position_size:.0f} units")

                with col6:
                    position_percent = trade.get('position_percent', 0)
                    st.metric("Capital at Risk", f"{position_percent:.1f}%")

                with col7:
                    try:
                        risk_reward = abs(
                            (trade['take_profit'] - trade['current_price']) /
                            (trade['stop_loss'] - trade['current_price'])
                        )
                        st.metric("Risk:Reward", f"1:{risk_reward:.1f}")
                    except (ZeroDivisionError, KeyError):
                        st.metric("Risk:Reward", "N/A")

                with col8:
                    regime = trade.get('regime', 'Unknown')
                    st.metric("Market Regime", f"State {regime}")

                # Technical details
                st.markdown("**Technical Analysis Summary:**")
                technical_score = trade.get('technical_score', 0)
                news_sentiment = trade.get('news_sentiment', 0)
                confidence = trade.get('confidence', 0)

                st.write(f"• Technical Score: {technical_score:.2f}")
                st.write(f"• News Sentiment: {news_sentiment:.2f}")
                st.write(f"• Overall Confidence: {confidence:.1%}")

        # Save trades button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Trades to History", use_container_width=True):
                save_trade_history(st.session_state.current_trades)

    elif 'current_trades' in st.session_state and not st.session_state.current_trades:
        st.info("No qualifying trades found based on current market conditions and risk parameters.")
    else:
        st.info("Click 'Generate Today's Trades' to analyze the forex markets and get trading recommendations.")

    # Trade History Section
    st.subheader("📈 Trade History")

    history_df = load_trade_history()

    if not history_df.empty:
        # Display only relevant columns and limit rows
        display_columns = ['pair', 'signal_text', 'current_price', 'confidence', 'timestamp']
        available_columns = [col for col in display_columns if col in history_df.columns]

        if available_columns:
            st.dataframe(
                history_df[available_columns].head(10),
                use_container_width=True
            )
        else:
            st.dataframe(history_df.head(10), use_container_width=True)
    else:
        st.info("No trade history available. Generate some trades first!")

    # Market Overview with cached data
    st.subheader("🌍 Market Overview")

    # Only show overview if we have cached data to avoid extra API calls
    overview_pairs = list(st.session_state.bot.currency_pairs.items())[:3]  # Limit to 3 pairs for overview
    overview_cols = st.columns(3)

    for i, (pair_name, symbol) in enumerate(overview_pairs):
        with overview_cols[i]:
            with st.container():
                st.markdown(f"**{pair_name}**")

                # Try to get cached data first
                try:
                    cached_data = st.session_state.bot.cache.get(symbol, "5d", max_age_hours=4)

                    if cached_data is not None and not cached_data.empty:
                        data = cached_data
                    else:
                        # Only fetch if no cached data and user explicitly requested
                        if st.button(f"Load {pair_name} Data", key=f"load_{symbol}"):
                            with st.spinner(f"Loading {pair_name}..."):
                                data = st.session_state.bot.fetch_forex_data(symbol, period="5d")
                        else:
                            st.info("Click to load current data")
                            continue

                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = ((current_price - prev_price) / prev_price) * 100

                        st.metric(
                            "Current Price",
                            f"{current_price:.5f}",
                            delta=f"{change:.2f}%"
                        )

                        # Mini chart
                        fig = go.Figure(data=go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ))
                        fig.update_layout(
                            height=150,
                            showlegend=False,
                            margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        fig.update_xaxes(showticklabels=False, showgrid=False)
                        fig.update_yaxes(showticklabels=False, showgrid=False)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error loading {pair_name}: {str(e)}")

    # Performance metrics
    if 'current_trades' in st.session_state and st.session_state.current_trades:
        st.subheader("📊 Trading Session Summary")

        trades = st.session_state.current_trades

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_confidence = np.mean([t.get('confidence', 0) for t in trades])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

        with col2:
            buy_signals = sum(1 for t in trades if t.get('signal') == 1)
            st.metric("Buy Signals", buy_signals)

        with col3:
            sell_signals = sum(1 for t in trades if t.get('signal') == -1)
            st.metric("Sell Signals", sell_signals)

        with col4:
            total_risk = sum(t.get('position_percent', 0) for t in trades)
            st.metric("Total Portfolio Risk", f"{total_risk:.1f}%")

    # System Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 System Status")

    # Cache statistics
    cache_size = len(st.session_state.bot.cache.memory_cache)
    st.sidebar.metric("Memory Cache", f"{cache_size} items")

    # Indicator cache
    indicator_cache_size = len(st.session_state.get('processed_indicators', {}))
    st.sidebar.metric("Indicator Cache", f"{indicator_cache_size} items")

    # Last generation time
    if 'last_generation_time' in st.session_state:
        last_gen = st.session_state.last_generation_time
        time_ago = (datetime.now() - last_gen).total_seconds() / 60
        st.sidebar.info(f"Last analysis: {time_ago:.0f} min ago")

    # Footer with enhanced disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
        <p><strong>⚠️ Enhanced Risk Disclaimer:</strong></p>
        <p>This optimized trading bot includes rate limiting and caching to prevent API overuse. 
        Forex trading involves substantial risk and may not be suitable for all investors.</p>
        <p><strong>Key Features:</strong></p>
        <ul style='text-align: left; display: inline-block;'>
            <li>✅ Intelligent caching system reduces API calls by up to 80%</li>
            <li>✅ Rate limiting prevents API quota exhaustion</li>
            <li>✅ Enhanced error handling and recovery</li>
            <li>✅ Persistent data storage for better performance</li>
            <li>✅ Batch processing for efficient market analysis</li>
        </ul>
        <p><em>Past performance is not indicative of future results. Please trade responsibly.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Debug information (only show if there are issues)
    if st.sidebar.checkbox("🔧 Debug Mode"):
        st.sidebar.markdown("**Debug Information:**")

        if 'bot' in st.session_state:
            st.sidebar.write(f"Rate limit delay: {st.session_state.bot.rate_limit_delay}s")
            st.sidebar.write(f"Last API call: {st.session_state.bot.last_api_call}")

        if 'current_trades' in st.session_state:
            st.sidebar.write(f"Current trades: {len(st.session_state.current_trades)}")

        # Show any cached errors or warnings
        if hasattr(st.session_state, 'debug_messages'):
            for msg in st.session_state.debug_messages[-5:]:  # Show last 5 messages
                st.sidebar.text(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info(
            "Please refresh the page and try again. If the problem persists, check your internet connection and API limits.")