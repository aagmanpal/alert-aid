"""
Script to retrain ML models with enhanced parameters
Run this to improve model accuracy with new configuration
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the ML predictor
try:
    from enhanced_main import DisasterPredictionModel
    
    logger.info("=" * 60)
    logger.info("RETRAINING ML MODELS WITH ENHANCED CONFIGURATION")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = DisasterPredictionModel()
    
    # Train models with enhanced parameters
    logger.info("\n🚀 Starting model training with enhanced parameters...")
    logger.info("   - Increased training samples: 25,000")
    logger.info("   - Enhanced model complexity (more estimators, better depth)")
    logger.info("   - Additional features: precipitation, vegetation, soil moisture, etc.")
    logger.info("   - Optimized hyperparameters for each disaster type\n")
    
    predictor.train_models()
    
    logger.info("\n✅ Model training completed successfully!")
    logger.info("\n📊 Updated Model Performance:")
    logger.info("-" * 60)
    
    for disaster_type, metrics in predictor.model_performance.items():
        logger.info(f"\n{disaster_type.upper()} Model:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Models saved and ready for production use!")
    logger.info("   Models location: backend/models/")
    logger.info("   Restart backend server to use updated models")
    logger.info("=" * 60)
    
except ImportError as e:
    logger.error(f"❌ Failed to import DisasterPredictionModel: {e}")
    logger.error("Make sure you're running this from the backend directory")
    logger.error("and all dependencies are installed (sklearn, numpy, etc.)")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Error during model training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
