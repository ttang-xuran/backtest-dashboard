#!/usr/bin/env python3
"""
Cointegration Testing Module for Statistical Arbitrage
Implements tests to validate mean-reverting relationships between asset pairs
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.regression.linear_model import OLS
except ImportError:
    print("Please install statsmodels: pip install statsmodels")
    exit(1)

class CointegrationTester:
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize Cointegration Tester
        
        Args:
            significance_level: Statistical significance level (default 0.05)
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
    def adf_test(self, series: np.array, critical_value: str = '5%') -> Tuple[bool, float, Dict]:
        """
        Augmented Dickey-Fuller test for stationarity
        
        Args:
            series: Time series data
            critical_value: Critical value level ('1%', '5%', '10%')
            
        Returns:
            (is_stationary, p_value, test_results)
        """
        try:
            result = adfuller(series, autolag='AIC')
            
            adf_stat = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # Check if series is stationary
            is_stationary = (adf_stat < critical_values[critical_value]) and (p_value < self.significance_level)
            
            test_results = {
                'adf_statistic': adf_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'used_lag': result[2],
                'n_observations': result[3]
            }
            
            self.logger.info(f"ADF Test - Stationary: {is_stationary}, P-value: {p_value:.6f}, "
                           f"ADF Stat: {adf_stat:.6f}, Critical Value ({critical_value}): {critical_values[critical_value]:.6f}")
            
            return is_stationary, p_value, test_results
            
        except Exception as e:
            self.logger.error(f"ADF test failed: {e}")
            return False, 1.0, {}
    
    def engle_granger_test(self, y: np.array, x: np.array) -> Tuple[bool, float, Dict]:
        """
        Engle-Granger cointegration test
        
        Args:
            y: First time series (dependent variable)
            x: Second time series (independent variable)
            
        Returns:
            (is_cointegrated, p_value, test_results)
        """
        try:
            # Perform cointegration test
            coint_stat, p_value, critical_values = coint(y, x)
            
            # Check if series are cointegrated
            is_cointegrated = p_value < self.significance_level
            
            # Calculate the cointegrating relationship
            model = OLS(y, np.column_stack([np.ones(len(x)), x])).fit()
            residuals = model.resid
            
            # Test residuals for stationarity
            residual_stationary, residual_p_value, _ = self.adf_test(residuals)
            
            test_results = {
                'cointegration_statistic': coint_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'cointegrating_vector': model.params,
                'residuals_stationary': residual_stationary,
                'residuals_p_value': residual_p_value,
                'r_squared': model.rsquared
            }
            
            self.logger.info(f"Engle-Granger Test - Cointegrated: {is_cointegrated}, "
                           f"P-value: {p_value:.6f}, Residuals Stationary: {residual_stationary}")
            
            return is_cointegrated, p_value, test_results
            
        except Exception as e:
            self.logger.error(f"Engle-Granger test failed: {e}")
            return False, 1.0, {}
    
    def johansen_test(self, data: np.array, det_order: int = 0, k_ar_diff: int = 1) -> Tuple[bool, Dict]:
        """
        Johansen cointegration test for multiple time series
        
        Args:
            data: Matrix of time series data (n_obs x n_vars)
            det_order: Deterministic term order (0=no constant, 1=constant, -1=constant+trend)
            k_ar_diff: Number of lags in differences
            
        Returns:
            (is_cointegrated, test_results)
        """
        try:
            # Perform Johansen test
            result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Check trace statistic against critical values
            trace_stats = result.lr1  # Trace statistics
            trace_critical = result.cvt  # Critical values for trace test
            
            # Check eigenvalue statistic against critical values  
            eigen_stats = result.lr2  # Eigenvalue statistics
            eigen_critical = result.cvm  # Critical values for eigenvalue test
            
            # Determine number of cointegrating relationships
            # Check at 5% significance level (index 1)
            significance_idx = 1  # 0=1%, 1=5%, 2=10%
            
            cointegrating_vectors = 0
            for i in range(len(trace_stats)):
                if trace_stats[i] > trace_critical[i, significance_idx]:
                    cointegrating_vectors = i + 1
            
            is_cointegrated = cointegrating_vectors > 0
            
            test_results = {
                'trace_statistics': trace_stats,
                'trace_critical_values': trace_critical,
                'eigenvalue_statistics': eigen_stats,
                'eigenvalue_critical_values': eigen_critical,
                'cointegrating_vectors_count': cointegrating_vectors,
                'cointegrating_vectors': result.evec,
                'eigenvalues': result.eig
            }
            
            self.logger.info(f"Johansen Test - Cointegrated: {is_cointegrated}, "
                           f"Cointegrating vectors: {cointegrating_vectors}")
            
            return is_cointegrated, test_results
            
        except Exception as e:
            self.logger.error(f"Johansen test failed: {e}")
            return False, {}
    
    def calculate_half_life(self, spread: np.array) -> float:
        """
        Calculate half-life of mean reversion for a spread
        
        Args:
            spread: Time series of spread values
            
        Returns:
            Half-life in time periods
        """
        try:
            # Calculate lagged spread and differences
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)
            
            # Run regression: Δspread_t = α + β * spread_{t-1} + ε_t
            model = OLS(spread_diff, np.column_stack([np.ones(len(spread_lag)), spread_lag])).fit()
            
            # Half-life calculation
            beta = model.params[1]
            if beta >= 0:
                return np.inf  # No mean reversion
            
            half_life = -np.log(2) / np.log(1 + beta)
            
            self.logger.info(f"Half-life calculated: {half_life:.2f} periods (β={beta:.6f})")
            
            return half_life
            
        except Exception as e:
            self.logger.error(f"Half-life calculation failed: {e}")
            return np.inf
    
    def comprehensive_cointegration_test(self, series1: np.array, series2: np.array, 
                                       series1_name: str = "Series1", 
                                       series2_name: str = "Series2") -> Dict:
        """
        Perform comprehensive cointegration analysis
        
        Args:
            series1: First time series
            series2: Second time series
            series1_name: Name of first series
            series2_name: Name of second series
            
        Returns:
            Dictionary with all test results
        """
        self.logger.info(f"Starting comprehensive cointegration analysis: {series1_name} vs {series2_name}")
        
        results = {
            'series_names': [series1_name, series2_name],
            'series_lengths': [len(series1), len(series2)],
            'timestamp': pd.Timestamp.now()
        }
        
        # 1. Test individual series for stationarity
        series1_stationary, series1_p, series1_adf = self.adf_test(series1)
        series2_stationary, series2_p, series2_adf = self.adf_test(series2)
        
        results['series1_stationarity'] = {
            'is_stationary': series1_stationary,
            'p_value': series1_p,
            'test_details': series1_adf
        }
        
        results['series2_stationarity'] = {
            'is_stationary': series2_stationary,
            'p_value': series2_p,
            'test_details': series2_adf
        }
        
        # 2. If both series are non-stationary, test for cointegration
        if not series1_stationary and not series2_stationary:
            self.logger.info("Both series are non-stationary, testing for cointegration...")
            
            # Engle-Granger test
            eg_cointegrated, eg_p_value, eg_results = self.engle_granger_test(series1, series2)
            results['engle_granger'] = {
                'is_cointegrated': eg_cointegrated,
                'p_value': eg_p_value,
                'test_details': eg_results
            }
            
            # Johansen test
            data_matrix = np.column_stack([series1, series2])
            johansen_cointegrated, johansen_results = self.johansen_test(data_matrix)
            results['johansen'] = {
                'is_cointegrated': johansen_cointegrated,
                'test_details': johansen_results
            }
            
            # Calculate spread and test its properties
            if eg_cointegrated:
                cointegrating_vector = eg_results.get('cointegrating_vector', [0, 1])
                spread = series1 - cointegrating_vector[1] * series2 - cointegrating_vector[0]
                
                # Test spread stationarity
                spread_stationary, spread_p, spread_adf = self.adf_test(spread)
                
                # Calculate half-life
                half_life = self.calculate_half_life(spread)
                
                results['spread_analysis'] = {
                    'spread_stationary': spread_stationary,
                    'spread_p_value': spread_p,
                    'spread_adf_details': spread_adf,
                    'half_life': half_life,
                    'cointegrating_vector': cointegrating_vector.tolist(),
                    'spread_mean': float(np.mean(spread)),
                    'spread_std': float(np.std(spread))
                }
        
        elif series1_stationary and series2_stationary:
            self.logger.info("Both series are stationary - no cointegration test needed")
            results['note'] = "Both series are stationary, cointegration testing not applicable"
        
        else:
            self.logger.warning("One series is stationary, one is not - mixed integration order")
            results['warning'] = "Mixed integration order detected - cointegration unlikely"
        
        # Overall assessment
        if 'engle_granger' in results and 'johansen' in results:
            eg_result = results['engle_granger']['is_cointegrated']
            johansen_result = results['johansen']['is_cointegrated']
            
            results['overall_assessment'] = {
                'is_cointegrated': eg_result and johansen_result,
                'engle_granger_agrees': eg_result,
                'johansen_agrees': johansen_result,
                'confidence': 'high' if eg_result and johansen_result else 'low',
                'suitable_for_pairs_trading': eg_result and johansen_result and 
                                            results.get('spread_analysis', {}).get('half_life', np.inf) < 100
            }
        
        self.logger.info(f"Cointegration analysis completed. Overall cointegrated: "
                        f"{results.get('overall_assessment', {}).get('is_cointegrated', False)}")
        
        return results
    
    def validate_trading_pair(self, btc_prices: list, eth_prices: list, 
                            min_observations: int = 20) -> Tuple[bool, str, Dict]:
        """
        Validate if BTC-ETH pair is suitable for statistical arbitrage
        
        Args:
            btc_prices: List of BTC prices
            eth_prices: List of ETH prices
            min_observations: Minimum number of observations required
            
        Returns:
            (is_suitable, reason, test_results)
        """
        # Check minimum observations
        if len(btc_prices) < min_observations or len(eth_prices) < min_observations:
            return False, f"Insufficient data: need {min_observations}, have {min(len(btc_prices), len(eth_prices))}", {}
        
        # Ensure equal length
        min_length = min(len(btc_prices), len(eth_prices))
        btc_prices = np.array(btc_prices[-min_length:])
        eth_prices = np.array(eth_prices[-min_length:])
        
        # Perform comprehensive cointegration test
        results = self.comprehensive_cointegration_test(btc_prices, eth_prices, "BTC", "ETH")
        
        # Determine suitability
        overall_assessment = results.get('overall_assessment', {})
        is_suitable = overall_assessment.get('suitable_for_pairs_trading', False)
        
        if is_suitable:
            half_life = results.get('spread_analysis', {}).get('half_life', np.inf)
            reason = f"Pair is suitable for trading. Half-life: {half_life:.2f} periods"
        else:
            if not overall_assessment.get('is_cointegrated', False):
                reason = "Series are not cointegrated - no stable long-term relationship"
            else:
                half_life = results.get('spread_analysis', {}).get('half_life', np.inf)
                if half_life > 100:
                    reason = f"Mean reversion too slow. Half-life: {half_life:.2f} periods"
                else:
                    reason = "Failed cointegration validation"
        
        return is_suitable, reason, results