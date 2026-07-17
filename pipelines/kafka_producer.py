#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.kafka_utils import NativeKafkaProducer, validate_native_setup, create_topic
from utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionEventGenerator:
    """Generate transaction events from real fraudTrain.csv dataset"""
    
    def __init__(self, seed: int = 42):
        config = load_config()
        data_path = config.get('data_paths', {}).get('raw_data', 'dataset/raw/fraudTrain.csv')
        if not os.path.isabs(data_path):
            data_path = os.path.join(project_root, data_path)
            
        logger.info(f"Loading raw data from {data_path}...")
        # Load first 10000 rows to ensure fast startup and low memory usage
        self.dataset = pd.read_csv(data_path, nrows=10000)
        self.dataset.dropna()

        if 'is_fraud' in self.dataset.columns:
            self.features = self.dataset.drop('is_fraud', axis=1)
            self.labels = self.dataset['is_fraud']
        else:
            self.features = self.dataset.copy()
            self.labels = None

        logger.info(f"Loaded {len(self.dataset)} transaction records!")
    
    def generate_event(self) -> Dict[str, Any]:
        """Generate single transaction event"""
        idx = random.randint(0, len(self.features) - 1)
        row = self.features.iloc[idx]

        event = {}
        for col, value in row.items():
            if pd.isna(value):
                event[col] = None 
            elif isinstance(value, (np.integer, np.int64)):
                event[col] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                event[col] = float(value)
            else:
                event[col] = str(value)
            
        event.update({
            'event_timestamp': datetime.utcnow().isoformat(),
            'event_id': f"evt_{idx}_{int(time.time())}",
            'true_fraud_label': int(self.labels.iloc[idx]) if self.labels is not None else None 
        })
        return event

    def generate_batch(self, num_events: int) -> List[Dict[str, Any]]:
        """Generate batch of events"""
        return [self.generate_event() for _ in range(num_events)]


class TransactionKafkaProducer:
    """Simplified Transaction Kafka Producer"""
    
    def __init__(self, enable_logging: bool = True):
        validation = validate_native_setup()
        if not validation['setup_valid']:
            raise RuntimeError("Kafka Setup is Invalid ...")

        self.producer = NativeKafkaProducer()
        self.generator = TransactionEventGenerator()
        self.enable_logging = enable_logging
    
    def _log_event(self, event: Dict[str, Any], success: bool, count: int):
        """Log event if logging enabled"""
        if not self.enable_logging:
            return
            
        status = "✅" if success else "❌"
        cc_num = str(event.get('cc_num', 'N/A'))
        category = str(event.get('category', 'N/A'))
        amount = str(event.get('amt', 'N/A'))
        
        print(f"{status} Event {count:3d}: Card {cc_num[:16]} | Cat: {category} | Amt: ${amount}")
    
    def setup_topic(self) -> bool:
        """Setup raw transactions topic"""
        config = load_config()
        kafka_config = config.get('kafka', {})
        topics = kafka_config.get('topics', {})
        raw_topic = topics.get('raw_transactions', 'raw_transactions')
        return create_topic(
            raw_topic, 
            partitions=1, 
            replication_factor=1
        )
    
    def produce_batch(self, topic: str = 'raw_transactions', num_events: int = 100) -> int:
        """Produce batch of events"""
        events = self.generator.generate_batch(num_events)
        successful = 0

        for i, event in enumerate(events):
            success = self.producer.send_message(
                topic=topic,
                message=event,
                key=str(event['cc_num'])   
            )

            if success:
                successful += 1 
                self._log_event(event, success, i+1)

        if self.enable_logging:
            print(f"Batch completed: {successful}/{num_events} events sent")
        return successful

    def produce_stream(self, topic: str = 'raw_transactions', 
                      rate: int = 1, duration: int = 300) -> int:
        """Produce streaming events"""
        start_time = time.time()
        total_events = 0
        successful = 0

        try:
            while time.time() - start_time < duration:
                batch_start = time.time()

                for _ in range(rate):
                    event = self.generator.generate_event()
    
                    success = self.producer.send_message(
                        topic=topic,
                        message=event,
                        key=str(event['cc_num'])   
                    )

                    total_events += 1 
                    if success:
                        successful += 1

                    self._log_event(event, success, total_events)

                sleep_time = max(0, 1 - (time.time() - batch_start))
                if sleep_time > 0: 
                    time.sleep(sleep_time)
    
            if self.enable_logging:
                print(f"Streaming completed: {successful}/{total_events} events sent")
            return successful

        except KeyboardInterrupt:
            logger.info("Streaming stopped by user.")
            return successful

    def close(self):
        """Close producer"""
        self.producer.close()


def main():
    """Main function"""
    config = load_config()
    kafka_config = config.get('kafka', {})
    topics = kafka_config.get('topics', {})
    default_topic = topics.get('raw_transactions', 'raw_transactions')

    parser = argparse.ArgumentParser(description="Kafka Producer for CC Fraud Detection ML Pipeline")
    parser.add_argument('--mode', choices=['streaming', 'batch'], default='streaming')
    parser.add_argument('--topic', default=default_topic)
    parser.add_argument('--rate', type=int, default=1, help='Events per second')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--num-events', type=int, default=100, help='Number of events')
    parser.add_argument('--setup-topics', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--quiet', action='store_true', help='Disable event logging')
    
    args = parser.parse_args()
    
    if args.validate:
        validation = validate_native_setup()
        if not validation['setup_valid']:
            logger.info("Kafka Setup is Invalid ...")
            return 1

    producer = TransactionKafkaProducer(enable_logging=not args.quiet)

    if args.setup_topics:
        if producer.setup_topic():
            logger.info("Topic Setup is Completed ...")
        else:
            logger.info("Topic Setup is Failed ...")

    if args.mode == 'streaming':
        producer.produce_stream(args.topic, args.rate, args.duration)
    else: 
        producer.produce_batch(args.topic, args.num_events)

    producer.close()
    return 0


if __name__ == "__main__":
    exit(main())
