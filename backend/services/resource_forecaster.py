import psutil
from collections import deque


class ResourceForecaster:
    """Small digital-twin proxy using real host measurements plus trend forecasting."""
    def __init__(self):
        self.cpu = deque(maxlen=8)
        self.ram = deque(maxlen=8)

    @staticmethod
    def _forecast(history):
        vals = list(history)
        if not vals:
            return 0.0
        if len(vals) == 1:
            return vals[-1]
        # weighted recent trend; lightweight surrogate for an LSTM resource forecaster
        diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        recent = diffs[-3:]
        trend = sum(recent) / len(recent)
        return max(0.0, min(100.0, vals[-1] + trend))

    def snapshot(self):
        cpu = float(psutil.cpu_percent(interval=0.15))
        ram = float(psutil.virtual_memory().percent)
        self.cpu.append(cpu)
        self.ram.append(ram)
        return {
            'cpu_now': round(cpu, 1),
            'ram_now': round(ram, 1),
            'cpu_forecast': round(self._forecast(self.cpu), 1),
            'ram_forecast': round(self._forecast(self.ram), 1),
            'source': 'Local machine telemetry',
            'forecast_method': 'Recent-trend digital-twin surrogate'
        }
