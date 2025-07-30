#!/usr/bin/env python3
"""
Parameter Optimization for Statistical Arbitrage Strategy
Finds optimal parameters using grid search and genetic algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import itertools
from multiprocessing import Pool, cpu_count
import json
import logging
from backtest_engine import StatArbBacktester

class ParameterOptimizer:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize parameter optimizer
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Optimization results storage
        self.optimization_results = []
        
    def grid_search_optimization(self, data: pd.DataFrame, 
                                param_ranges: Dict,
                                objective: str = "sharpe_ratio",
                                n_jobs: int = None) -> Dict:
        """
        Perform grid search optimization
        
        Args:
            data: Historical price data
            param_ranges: Dictionary of parameter ranges to test
            objective: Optimization objective ("sharpe_ratio", "total_return", "calmar_ratio")
            n_jobs: Number of parallel jobs (default: CPU count)
            
        Returns:
            Best parameters and results
        """
        self.logger.info("Starting grid search optimization...")
        
        if n_jobs is None:
            n_jobs = min(cpu_count(), 8)
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations using {n_jobs} processes")
        
        # Prepare arguments for parallel processing
        args_list = [(data, dict(zip(param_names, combo)), objective) 
                     for combo in param_combinations]
        
        # Run optimization in parallel
        with Pool(n_jobs) as pool:
            results = pool.starmap(self._evaluate_parameters, args_list)
        
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        
        # Save all results
        self.optimization_results = results
        
        self.logger.info(f"Optimization completed. Best {objective}: {best_result['score']:.4f}")
        
        return best_result
    
    def _evaluate_parameters(self, data: pd.DataFrame, params: Dict, objective: str) -> Dict:
        """
        Evaluate a single parameter combination
        
        Args:
            data: Historical price data
            params: Parameter combination to test
            objective: Optimization objective
            
        Returns:
            Evaluation result
        """
        try:
            # Create temporary config with test parameters
            temp_config = self._create_temp_config(params)
            
            # Run backtest with these parameters
            backtester = StatArbBacktester()
            backtester.config = temp_config
            backtester.position_size = temp_config.get("position_size", 0.25)
            backtester.z_score_entry = temp_config.get("z_score_entry", 1.5)
            backtester.z_score_exit = temp_config.get("z_score_exit", 0.3)
            backtester.lookback_period = temp_config.get("lookback_period", 20)
            backtester.stop_loss_pct = temp_config.get("stop_loss_pct", 0.02)
            
            results = backtester.run_backtest(data)
            
            # Calculate objective score
            score = self._calculate_objective_score(results, objective)
            
            return {
                'parameters': params,
                'score': score,
                'results': results,
                'objective': objective
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters {params}: {e}")
            return {
                'parameters': params,
                'score': -np.inf,
                'results': {},
                'objective': objective,
                'error': str(e)
            }
    
    def _create_temp_config(self, params: Dict) -> Dict:
        """Create temporary configuration with test parameters"""
        # Load base config
        try:
            with open(self.config_path, 'r') as f:
                base_config = json.load(f)
        except:
            base_config = {}
        
        # Update with test parameters
        temp_config = base_config.copy()
        temp_config.update(params)
        
        return temp_config
    
    def _calculate_objective_score(self, results: Dict, objective: str) -> float:
        """Calculate objective score from backtest results"""
        if not results or 'total_return' not in results:
            return -np.inf
        
        if objective == "sharpe_ratio":
            return results.get('sharpe_ratio', -np.inf)
        elif objective == "total_return":
            return results.get('total_return', -np.inf)
        elif objective == "calmar_ratio":
            # Calmar ratio = Annual return / Max drawdown
            annual_return = results.get('total_return', 0) * 365
            max_drawdown = abs(results.get('max_drawdown', 1))
            return annual_return / max_drawdown if max_drawdown > 0 else -np.inf
        elif objective == "profit_factor":
            # Simplified profit factor
            total_return = results.get('total_return', 0)
            max_drawdown = abs(results.get('max_drawdown', 1))
            return (1 + total_return) / (1 + max_drawdown) if max_drawdown > 0 else -np.inf
        else:
            return results.get(objective, -np.inf)
    
    def walk_forward_optimization(self, data: pd.DataFrame,
                                 param_ranges: Dict,
                                 train_days: int = 15,
                                 test_days: int = 5,
                                 objective: str = "sharpe_ratio") -> List[Dict]:
        """
        Perform walk-forward optimization
        
        Args:
            data: Full historical dataset
            param_ranges: Parameter ranges to optimize
            train_days: Training period length
            test_days: Testing period length
            objective: Optimization objective
            
        Returns:
            List of walk-forward results
        """
        self.logger.info("Starting walk-forward optimization...")
        
        results = []
        total_days = len(data) // (24 * 60)  # Assuming minute data
        
        for start_day in range(0, total_days - train_days - test_days, test_days):
            # Define train and test periods
            train_start = start_day * 24 * 60
            train_end = (start_day + train_days) * 24 * 60
            test_start = train_end
            test_end = test_start + test_days * 24 * 60
            
            if test_end > len(data):
                break
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            self.logger.info(f"Period {len(results)+1}: Training on days {start_day}-{start_day+train_days}, "
                           f"testing on days {start_day+train_days}-{start_day+train_days+test_days}")
            
            # Optimize on training data
            best_params = self.grid_search_optimization(
                train_data, param_ranges, objective, n_jobs=4
            )
            
            # Test on out-of-sample data
            backtester = StatArbBacktester()
            temp_config = self._create_temp_config(best_params['parameters'])
            backtester.config = temp_config
            backtester.position_size = temp_config.get("position_size", 0.25)
            backtester.z_score_entry = temp_config.get("z_score_entry", 1.5)
            backtester.z_score_exit = temp_config.get("z_score_exit", 0.3)
            backtester.lookback_period = temp_config.get("lookback_period", 20)
            backtester.stop_loss_pct = temp_config.get("stop_loss_pct", 0.02)
            
            test_results = backtester.run_backtest(test_data)
            
            results.append({
                'period': len(results) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params['parameters'],
                'train_score': best_params['score'],
                'test_results': test_results,
                'test_score': self._calculate_objective_score(test_results, objective)
            })
        
        return results
    
    def analyze_parameter_sensitivity(self, data: pd.DataFrame, 
                                    base_params: Dict,
                                    param_to_test: str,
                                    test_range: List,
                                    objective: str = "sharpe_ratio") -> Dict:
        """
        Analyze sensitivity of a single parameter
        
        Args:
            data: Historical price data
            base_params: Base parameter configuration
            param_to_test: Parameter name to analyze
            test_range: Range of values to test
            objective: Optimization objective
            
        Returns:
            Sensitivity analysis results
        """
        self.logger.info(f"Analyzing sensitivity of parameter: {param_to_test}")
        
        results = []
        
        for value in test_range:
            test_params = base_params.copy()
            test_params[param_to_test] = value
            
            result = self._evaluate_parameters(data, test_params, objective)
            results.append({
                'parameter_value': value,
                'score': result['score'],
                'results': result['results']
            })
        
        return {
            'parameter': param_to_test,
            'sensitivity_results': results,
            'base_params': base_params,
            'objective': objective
        }
    
    def save_optimization_results(self, results: Dict, filename: str = "optimization_results.json"):
        """Save optimization results to file"""
        try:
            # Convert numpy types to native Python types for JSON serialization
            results_serializable = self._make_serializable(results)
            
            with open(filename, 'w') as f:
                json.dump(results_serializable, f, indent=2, default=str)
            
            self.logger.info(f"Optimization results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def plot_optimization_results(self, results: List[Dict], param_name: str):
        """Plot optimization results for visualization"""
        import matplotlib.pyplot as plt
        
        if not results:
            self.logger.warning("No results to plot")
            return
        
        # Extract parameter values and scores
        param_values = [r['parameters'].get(param_name, 0) for r in results]
        scores = [r['score'] for r in results if r['score'] != -np.inf]
        valid_params = [p for i, p in enumerate(param_values) if results[i]['score'] != -np.inf]
        
        if not scores:
            self.logger.warning("No valid scores to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_params, scores, alpha=0.7)
        plt.xlabel(param_name)
        plt.ylabel('Objective Score')
        plt.title(f'Parameter Optimization: {param_name}')
        plt.grid(True, alpha=0.3)
        
        # Highlight best result
        best_idx = np.argmax(scores)
        plt.scatter(valid_params[best_idx], scores[best_idx], 
                   color='red', s=100, label=f'Best: {valid_params[best_idx]:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'optimization_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_sample_optimization():
    """Run a sample optimization"""
    print("Statistical Arbitrage Parameter Optimization")
    print("="*50)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Generate sample data
    backtester = StatArbBacktester()
    data = backtester.generate_sample_data(days=10)  # 10 days for quick test
    
    # Define parameter ranges to optimize
    param_ranges = {
        'z_score_entry': [1.0, 1.5, 2.0, 2.5],
        'z_score_exit': [0.1, 0.3, 0.5],
        'lookback_period': [10, 15, 20, 25],
        'position_size': [0.1, 0.15, 0.2, 0.25]
    }
    
    # Run optimization
    best_result = optimizer.grid_search_optimization(
        data, param_ranges, objective="sharpe_ratio"
    )
    
    # Display results
    print(f"\nBest Parameters:")
    for param, value in best_result['parameters'].items():
        print(f"  {param}: {value}")
    
    print(f"\nBest Sharpe Ratio: {best_result['score']:.4f}")
    
    # Save results
    optimizer.save_optimization_results(best_result, "best_params.json")
    
    return best_result


if __name__ == "__main__":
    # Install missing dependencies first
    print("Note: Run 'pip install matplotlib seaborn' if not already installed")
    
    results = run_sample_optimization()