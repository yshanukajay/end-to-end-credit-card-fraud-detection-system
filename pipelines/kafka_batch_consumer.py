#!/usr/bin/env python3
"""
Simplified Kafka Consumer with ML Predictions for Credit Card Fraud Detection
Processes transaction events with real-time ML inference
"""

import json
import logging
import argparse
import os
import sys
import time
from typing import Dict, Any
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from confluent_kafka import Consumer, Producer, KafkaError
from src.model_inference import ModelInference
from utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLKafkaConsumer:
    """Simplified ML Kafka Consumer for Credit Card Fraud Detection"""
    
    def __init__(self):
        self.model = None
        self.config = load_config()
        
        # Load Kafka topics
        kafka_config = self.config.get('kafka', {})
        topics = kafka_config.get('topics', {})
        self.input_topic = topics.get('raw_transactions', 'raw_transactions')
        self.output_topic = topics.get('fraud_predictions', 'fraud_predictions')
        
    def initialize(self):
        """Initialize ML model"""
        try:
            model_cfg = self.config.get('model', {})
            model_path = model_cfg.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl')
            self.model = ModelInference(model_path=model_path, use_spark=False)
            
            # Load encoders
            encoders_dir = "artifacts/encode"
            if os.path.exists(encoders_dir):
                self.model.load_encoders(encoders_dir)
                logger.info("✅ ML model and encoders loaded successfully")
            
            return True
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def extract_transaction_data(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate transaction data"""
        data = message_data.get('data', message_data)
        
        # Required fields for credit card transaction matching features
        return {
            'trans_date_trans_time': data.get('trans_date_trans_time', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')),
            'cc_num': data.get('cc_num', 0),
            'amt': data.get('amt', 0.0),
            'category': data.get('category', 'misc_pos'),
            'gender': data.get('gender', 'F'),
            'lat': data.get('lat', 40.0),
            'long': data.get('long', -80.0),
            'city_pop': data.get('city_pop', 1000.0),
            'dob': data.get('dob', '1990-01-01'),
            'merch_lat': data.get('merch_lat', 40.0),
            'merch_long': data.get('merch_long', -80.0),
            'velocity_last_24h': data.get('velocity_last_24h', 0.0)
        }
    
    def process_batch(self, max_messages: int = 1000, timeout: int = 10, 
                     group_id: str = None) -> int:
        """Process batch of messages with ML predictions"""
        
        if group_id is None:
            group_id = f"batch_consumer_{int(time.time())}"
        
        consumer_config = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': group_id,
            'auto.offset.reset': 'earliest' if 'batch_' in group_id else 'latest',
            'enable.auto.commit': True
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([self.input_topic])
        
        # Collect messages
        messages = []
        start_time = time.time()
        
        while len(messages) < max_messages and (time.time() - start_time) < timeout:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    break
                continue
            
            try:
                message_data = json.loads(msg.value().decode('utf-8'))
                messages.append(message_data)
            except json.JSONDecodeError:
                continue
        
        consumer.close()
        
        if not messages:
            logger.warning("⚠️ No messages to process")
            return 0
        
        # Process with ML
        logger.info(f"📥 Processing {len(messages)} messages with ML")
        
        # Setup producer for results
        producer = Producer({'bootstrap.servers': 'localhost:9092'})
        processed = 0
        
        print(f"\n📊 ML PREDICTIONS (Credit Card Fraud Detection)")
        print("=" * 80)
        print(f"{'Status':<6} | {'Card Number':<10} | {'Category':<10} | {'Action':<15} | {'Probability':<12} | {'Confidence':<10}")
        print("-" * 80)
        
        for i, message_data in enumerate(messages):
            try:
                # Extract transaction data
                trans_data = self.extract_transaction_data(message_data)
                cc_num = str(trans_data.get('cc_num', 'N/A'))
                category = str(trans_data.get('category', 'N/A'))[:10]
                
                # Make prediction
                prediction = self.model.predict(trans_data)
                pred_label = prediction.get('Prediction', 0)
                status = prediction.get('Status', 'Unknown')
                action = prediction.get('Action', 'Unknown')
                probability = prediction.get('Probability', 0.0)
                confidence = prediction.get('Confidence', '0%')
                
                # Display result
                pred_emoji = "🚨" if pred_label == 1 else "✅"
                print(f"  {pred_emoji}   | {cc_num[:10]:10s} | {category:10s} | {action:15s} | {probability:<12.4f} | {confidence:10s}")
                
                # Send result
                result = {
                    'cc_num': cc_num,
                    'original_data': trans_data,
                    'prediction': prediction,
                    'processed_at': datetime.now().isoformat(),
                    'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
                
                producer.produce(
                    topic=self.output_topic,
                    key=cc_num,
                    value=json.dumps(result, default=str)
                )
                
                processed += 1
                
            except Exception as e:
                print(f"  ❌   | ERROR      | ERROR      | FAILED          | ERROR        | ERROR")
                logger.error(f"Error processing message {i}: {str(e)}")
        
        producer.flush()
        
        print("-" * 80)
        print(f"✅ Completed: {processed}/{len(messages)} predictions")
        print("=" * 80)
        
        logger.info(f"🎉 Processed {processed} messages successfully")
        return processed
    
    def run_continuous(self, poll_interval: int = 3, show_progress: bool = True):
        """Run continuous processing"""
        logger.info("🔄 Starting continuous ML processing")
        logger.info("🛑 Press Ctrl+C to stop")
        
        total_processed = 0
        
        try:
            while True:
                if show_progress:
                    print(f"\n📡 Checking for new messages... (Total: {total_processed})")
                
                # Process new messages
                processed = self.process_batch(
                    max_messages=50,
                    timeout=poll_interval,
                    group_id='continuous_ml_consumer'
                )
                
                if processed > 0:
                    total_processed += processed
                    print(f"✅ Processed {processed} new messages (Total: {total_processed})")
                else:
                    if show_progress:
                        print("⏳ No new messages - waiting...")
                    else:
                        print(".", end="", flush=True)
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            logger.info(f"🛑 Continuous processing stopped (Total: {total_processed})")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Kafka Consumer with ML Predictions")
    parser.add_argument('--max-messages', type=int, default=1000)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--poll-interval', type=int, default=3)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting Kafka ML Consumer")
        
        consumer = MLKafkaConsumer()
        if not consumer.initialize():
            return 1
        
        if args.continuous:
            consumer.run_continuous(args.poll_interval, not args.quiet)
        else:
            processed = consumer.process_batch(args.max_messages, args.timeout)
            return 0 if processed > 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Consumer failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())