"""
ExamGuard AI (PRJ 137) - Configurable Event & Risk Scoring Engine
Calculates the Proctoring Risk Indicator based on configurable event weights.
Outputs explainable event scores and risk levels (LOW, MODERATE, HIGH, CRITICAL).
"""

import os
import yaml

class RiskEngine:
    def __init__(self, config_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if config_path is None:
            # Check configs/proctoring.yaml then config/proctoring.yaml
            p1 = os.path.join(base_dir, 'configs', 'proctoring.yaml')
            p2 = os.path.join(base_dir, 'config', 'proctoring.yaml')
            config_path = p1 if os.path.exists(p1) else p2
            
        self.weights = {
            'FACE_MISSING': 35.0,
            'MULTIPLE_FACES': 45.0,
            'IDENTITY_MISMATCH': 50.0,
            'PHONE_DETECTED': 40.0,
            'PROHIBITED_OBJECT_DETECTED': 30.0,
            'PERSON_DETECTED': 25.0,
            'LOOKING_AWAY': 15.0,
            'HEAD_POSE_ANOMALY': 20.0,
            'BROWSER_TAB_SWITCH': 20.0,
            'CLIPBOARD_COPY_PASTE': 15.0,
            'SCREENSHOT_INTERCEPT': 25.0
        }
        
        self.thresholds = {
            'LOW_MAX': 24.0,
            'MODERATE_MAX': 49.0,
            'HIGH_MAX': 74.0,
            'CRITICAL_MIN': 75.0
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
                if 'event_weights' in cfg:
                    self.weights.update(cfg['event_weights'])
                if 'risk_classification_thresholds' in cfg:
                    self.thresholds.update(cfg['risk_classification_thresholds'])

    def calculate_risk(self, active_conditions, event_history=None):
        """
        Calculates Proctoring Risk Indicator (0 - 100).
        active_conditions: dict of {condition_name: boolean_or_count}
        Returns:
            dict: {
                'proctoring_risk_indicator': int,
                'candidate_integrity_index': int,
                'risk_level': str,
                'contributors': list of dicts
            }
        """
        score = 5.0 # Baseline benign presence
        contributors = []
        
        for cond, val in active_conditions.items():
            weight = self.weights.get(cond, 10.0)
            if isinstance(val, bool) and val:
                score += weight
                contributors.append({'condition': cond, 'delta': int(weight)})
            elif isinstance(val, (int, float)) and val > 0:
                delta = weight * val
                score += delta
                contributors.append({'condition': f"{cond} ({val}x)", 'delta': int(delta)})
                
        # Clamp to 0..99
        final_score = int(min(99, max(0, score)))
        integrity_index = max(0, 100 - final_score)
        
        # Risk Level Classification
        if final_score >= self.thresholds['CRITICAL_MIN']:
            risk_level = 'CRITICAL'
        elif final_score > self.thresholds['MODERATE_MAX']:
            risk_level = 'HIGH'
        elif final_score > self.thresholds['LOW_MAX']:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
            
        return {
            'proctoring_risk_indicator': final_score,
            'candidate_integrity_index': integrity_index,
            'risk_level': risk_level,
            'contributors': contributors
        }
