from backend.config import CONFIDENCE_THRESHOLD, HIGH_RISK_THRESHOLD, CPU_LIMIT, RAM_LIMIT


class FogOrchestrator:
    MODEL_COST = {'tiny_ids': 1, 'decision_tree': 2, 'random_forest': 3}

    def decide(self, results, resources, risk, metrics=None):
        metrics = metrics or {}
        best_edge = max(results, key=lambda x: x['confidence'])

        votes = [r['prediction'] for r in results]
        disagreement = len(set(votes)) > 1
        low_confidence = best_edge['confidence'] < CONFIDENCE_THRESHOLD
        escalated = low_confidence or disagreement or risk >= HIGH_RISK_THRESHOLD

        if not escalated:
            return {
                'escalated': False,
                'gate_reason': 'Edge models agree with sufficient confidence.',
                'selected_model': best_edge['key'],
                'selected_model_name': best_edge['model'],
                'reason': 'Handled locally because confidence and model agreement are sufficient.',
                'action': 'Local alert' if best_edge['prediction'] == 'Attack' else 'Allow / monitor'
            }

        constrained = (
            resources['cpu_forecast'] > CPU_LIMIT or
            resources['ram_forecast'] > RAM_LIMIT
        )

        if constrained:
            candidates = ['decision_tree', 'tiny_ids']
            gate_reason = 'Fog escalation triggered; resource forecast is constrained.'
        else:
            candidates = ['random_forest', 'decision_tree', 'tiny_ids']
            gate_reason = 'Fog escalation triggered by uncertainty, disagreement, or high risk.'

        result_map = {r['key']: r for r in results}
        def utility(key):
            confidence = result_map[key]['confidence']
            f1 = metrics.get(key, {}).get('f1', 0.5)
            resource_penalty = self.MODEL_COST[key] * (0.04 if constrained else 0.01)
            risk_bonus = 0.05 if (risk >= HIGH_RISK_THRESHOLD and key == 'random_forest') else 0
            return 0.5 * confidence + 0.45 * f1 + risk_bonus - resource_penalty

        selected = max(candidates, key=utility)
        selected_result = result_map[selected]
        reason = (
            f"Selected {selected_result['model']} from confidence, validation F1, "
            f"attack risk ({risk:.1f}/100), and predicted resource availability."
        )
        return {
            'escalated': True,
            'gate_reason': gate_reason,
            'selected_model': selected,
            'selected_model_name': selected_result['model'],
            'reason': reason,
            'action': 'Block + alert' if selected_result['prediction'] == 'Attack' else 'Monitor'
        }
