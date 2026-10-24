from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR/'models'
DATA_DIR=BASE_DIR/'data'
FRONTEND_DIR=BASE_DIR/'frontend'
CONFIDENCE_THRESHOLD=0.72
HIGH_RISK_THRESHOLD=70
CPU_LIMIT=78
RAM_LIMIT=80
FEATURES=['packet_rate','failed_logins','unique_ports','avg_packet_size','connection_duration','syn_ratio']
