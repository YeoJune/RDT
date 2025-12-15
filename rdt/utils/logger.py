"""Simple CSV logger for training metrics"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class CSVLogger:
    """Simple CSV logger for tracking training metrics"""
    
    def __init__(self, log_dir: str, filename: Optional[str] = None):
        """
        Args:
            log_dir: Directory to save logs
            filename: Log filename (auto-generated if None)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'train_{timestamp}.csv'
        
        self.log_path = self.log_dir / filename
        self.writer = None
        self.file = None
        self.fieldnames = None
        
    def log(self, metrics: Dict[str, float]):
        """Log metrics to CSV"""
        if self.writer is None:
            # Initialize on first call
            self.fieldnames = list(metrics.keys())
            self.file = open(self.log_path, 'w', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            print(f"Logging to: {self.log_path}")
        
        self.writer.writerow(metrics)
        self.file.flush()
    
    def close(self):
        """Close the log file"""
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None
    
    def __del__(self):
        self.close()
