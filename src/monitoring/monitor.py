import os
import logging
import joblib
import pandas as pd
import numpy as np
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Tuple
import mlflow

from src.utils.utils import load_processed_data
from src.training.train import evaluate

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance on test data and trigger retraining when needed"""
    
    def __init__(self, 
                 model_path: str = "shared/models/gbr.joblib",
                 processed_dir: str = "shared/data/processed"):
        self.model_path = model_path
        self.processed_dir = processed_dir
        
        # Baseline metrics from undrifted data
        self.baseline_metrics = {
            "rmse": 0.24663,
            "mae": 0.33386,
            "r2": 0.812029
        }
        
        # Acceptable ranges based on ~6 months of 3% annual inflation
        # Inflation multiplier for 6 months: exp(ln(1.03) * 0.5) ≈ 1.0148
        # This typically increases RMSE/MAE by ~5-10% and decreases R2 by ~2-3%
        self.acceptable_thresholds = {
            "rmse": {
                "min": self.baseline_metrics["rmse"] * 0.9,   # 10% better (unlikely but possible)
                "max": self.baseline_metrics["rmse"] * 1.15   # 15% worse (degradation threshold)
            },
            "mae": {
                "min": self.baseline_metrics["mae"] * 0.9,
                "max": self.baseline_metrics["mae"] * 1.15
            },
            "r2": {
                "min": self.baseline_metrics["r2"] * 0.92,    # 8% relative decrease
                "max": self.baseline_metrics["r2"] * 1.05     # 5% relative increase (cap)
            }
        }
        
        logger.info(f"Monitoring thresholds: {self.acceptable_thresholds}")
    
    def load_model(self) -> Any:
        """Load the trained model from shared storage"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        saved = joblib.load(self.model_path)
        
        # Handle both formats: {"model": ..., "params": ...} or direct model
        if isinstance(saved, dict) and "model" in saved:
            model = saved["model"]
            logger.info(f"Loaded model with params: {saved.get('params', 'unknown')}")
        else:
            model = saved
            logger.info("Loaded direct model object")
            
        return model
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data from processed directory"""
        logger.info(f"Loading test data from {self.processed_dir}")
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(self.processed_dir)
            logger.info(f"Test data loaded: X_test shape {X_test.shape}, y_test length {len(y_test)}")
            return X_test, y_test
        except FileNotFoundError as e:
            logger.error(f"Could not load test data: {e}")
            raise
    
    def evaluate_model_performance(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test data using training.train's evaluate function"""
        logger.info("Evaluating model performance on test data...")
        
        # Use the same evaluate function from training module
        metrics = evaluate(model, X_test, y_test)
        
        logger.info(f"Current model metrics: RMSE={metrics['rmse']:.5f}, "
                   f"MAE={metrics['mae']:.5f}, R2={metrics['r2']:.6f}")
        
        return metrics
    
    def check_performance_degradation(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if current metrics are within acceptable thresholds"""
        degradation_report = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "baseline_metrics": self.baseline_metrics,
            "thresholds": self.acceptable_thresholds,
            "metric_status": {},
            "overall_degraded": False,
            "degraded_metrics": []
        }
        
        # Check each metric against thresholds
        for metric_name, current_value in current_metrics.items():
            thresholds = self.acceptable_thresholds[metric_name]
            
            is_degraded = (
                current_value < thresholds["min"] or 
                current_value > thresholds["max"]
            )
            
            degradation_report["metric_status"][metric_name] = {
                "current": float(current_value),
                "baseline": float(self.baseline_metrics[metric_name]),
                "min_threshold": float(thresholds["min"]),
                "max_threshold": float(thresholds["max"]),
                "is_degraded": is_degraded,
                "deviation_pct": float((current_value - self.baseline_metrics[metric_name]) / self.baseline_metrics[metric_name] * 100)
            }
            
            if is_degraded:
                degradation_report["degraded_metrics"].append(metric_name)
        
        # Overall degradation if any metric is outside thresholds
        degradation_report["overall_degraded"] = len(degradation_report["degraded_metrics"]) > 0
        
        # Log summary
        if degradation_report["overall_degraded"]:
            logger.warning(f"Performance degradation detected in: {degradation_report['degraded_metrics']}")
        else:
            logger.info("Model performance within acceptable thresholds")
        
        return degradation_report
    
    def save_monitoring_report(self, degradation_report: Dict[str, Any]) -> str:
        """Save monitoring report to shared storage"""
        report_path = "shared/monitoring/model_performance_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(degradation_report, f, indent=2)
        
        logger.info(f"Monitoring report saved to {report_path}")
        return report_path
    
    def trigger_retraining(self) -> bool:
        """Trigger model retraining by calling training module"""
        logger.info("Triggering model retraining due to performance degradation...")
        
        try:
            # Run training module
            result = subprocess.run([
                sys.executable, '-m', 'src.training.train'
            ], capture_output=True, text=True, check=True)
            
            logger.info("Model retraining completed successfully")
            logger.info(f"Training output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Model retraining failed: {e}")
            logger.error(f"Training stderr: {e.stderr}")
            return False
    
    def log_to_mlflow(self, degradation_report: Dict[str, Any], report_path: str):
        """Log monitoring results to MLflow"""
        try:
            logger.info("Logging monitoring results to MLflow...")
            
            mlflow.set_experiment("gradient_boosted_regression")
            with mlflow.start_run(run_name="model_monitoring", nested=True):
                # Log current metrics
                for metric_name, metric_value in degradation_report["current_metrics"].items():
                    mlflow.log_metric(f"current_{metric_name}", metric_value)
                
                # Log baseline metrics for comparison
                for metric_name, baseline_value in degradation_report["baseline_metrics"].items():
                    mlflow.log_metric(f"baseline_{metric_name}", baseline_value)
                
                # Log degradation info
                mlflow.log_param("overall_degraded", degradation_report["overall_degraded"])
                mlflow.log_param("degraded_metrics", ",".join(degradation_report["degraded_metrics"]))
                mlflow.log_metric("degraded_metrics_count", len(degradation_report["degraded_metrics"]))
                
                # Log detailed metric status
                for metric_name, status in degradation_report["metric_status"].items():
                    mlflow.log_metric(f"{metric_name}_deviation_pct", status["deviation_pct"])
                    mlflow.log_param(f"{metric_name}_degraded", status["is_degraded"])
                
                # Log monitoring report as artifact
                mlflow.log_artifact(report_path, artifact_path="monitoring")
            
            logger.info("MLflow logging completed")
            
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    def run_monitoring(self) -> Dict[str, Any]:
        """Run complete monitoring workflow"""
        logger.info("Starting model performance monitoring...")
        
        result = {
            "status": "success",
            "degraded": False,
            "retraining_triggered": False,
            "retraining_success": False
        }
        
        try:
            # Step 1: Load model and test data
            model = self.load_model()
            X_test, y_test = self.load_test_data()
            
            # Step 2: Evaluate current performance
            current_metrics = self.evaluate_model_performance(model, X_test, y_test)
            
            # Step 3: Check for degradation
            degradation_report = self.check_performance_degradation(current_metrics)
            
            # Step 4: Save report
            report_path = self.save_monitoring_report(degradation_report)
            
            # Step 5: Log to MLflow
            self.log_to_mlflow(degradation_report, report_path)
            
            # Step 6: Trigger retraining if needed
            result["degraded"] = degradation_report["overall_degraded"]
            
            if degradation_report["overall_degraded"]:
                logger.info("Performance degradation detected - triggering retraining")
                result["retraining_triggered"] = True
                result["retraining_success"] = self.trigger_retraining()
            else:
                logger.info("Model performance acceptable - no retraining needed")
            
            # Update final result
            result.update({
                "current_metrics": current_metrics,
                "degraded_metrics": degradation_report["degraded_metrics"],
                "report_path": report_path
            })
            
            logger.info("Model monitoring completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Model monitoring failed: {e}")
            result.update({
                "status": "failed",
                "error": str(e)
            })
            return result

def main():
    """
    Main monitoring entry point - can be run standalone or as part of pipeline
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-service:5000')
    mlflow.set_tracking_uri(mlflow_uri)    
    mlflow.set_experiment("gradient_boosted_regression")
    
    # Determine if we should create a new run or nest under existing
    if mlflow.active_run() is None:
        run_context = mlflow.start_run(run_name="model_monitoring")
    else:
        run_context = mlflow.start_run(run_name="model_monitoring", nested=True)
    
    with run_context:
        monitor = ModelMonitor()
        result = monitor.run_monitoring()
        
        # Log high-level results
        mlflow.log_param("monitoring_status", result["status"])
        if result["status"] == "success":
            mlflow.log_param("performance_degraded", result["degraded"])
            mlflow.log_param("retraining_triggered", result["retraining_triggered"])
            if result["retraining_triggered"]:
                mlflow.log_param("retraining_success", result["retraining_success"])
        
        # Return boolean for pipeline decisions
        # Return True if model needs retraining (degraded performance)
        return result["degraded"] if result["status"] == "success" else True

if __name__ == "__main__":
    needs_retraining = main()
    # Exit with code 1 if retraining is needed, 0 if not
    exit(1 if needs_retraining else 0)