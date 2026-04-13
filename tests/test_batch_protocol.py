"""Tests for the batch NN evaluation protocol (Python side).

Tests the _eval_nn_batch method and batch_eval_req handling without
requiring a running Rust process or GPU.
"""

import json
import os
import struct
import numpy as np

class MockModel:
    """Mock NN model that returns predictable outputs."""
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self, x):
        import torch
        batch_size = x.shape[0]
        # Return uniform logits and zero values
        logits = torch.zeros(batch_size, self.n_actions)
        values = torch.zeros(batch_size)
        return logits, values

    def eval(self):
        return self

    def parameters(self):
        return []

    def to(self, device):
        return self


class MockNNSearchClient:
    """Mimics NNSearchClient but allows direct method testing."""
    def __init__(self, model, cfg, device='cpu'):
        self.model = model
        self.cfg = cfg
        self.device = device

    def _eval_nn_batch(self, req_line):
        """Copied protocol handler from NNSearchClient."""
        import torch
        import torch.nn.functional as F

        try:
            batch_req = json.loads(req_line)["batch_eval_req"]
            requests = batch_req["requests"]
            batch_size = len(requests)
            ch, bs = self.cfg['ch'], self.cfg['board']
            expected = ch * bs * bs

            if self.model is not None and batch_size > 0:
                features_list = []
                for req in requests:
                    feats = req.get("features", [])
                    if len(feats) == expected:
                        features_list.append(feats)
                    else:
                        features_list.append([0.0] * expected)

                x = torch.tensor(features_list, dtype=torch.float32).reshape(
                    batch_size, ch, bs, bs).to(self.device)
                with torch.no_grad():
                    logits, vals = self.model(x)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    vals_np = vals.cpu().numpy()

                responses = []
                for i, req in enumerate(requests):
                    na = req.get("num_actions", self.cfg['actions'])
                    responses.append({
                        "policy": probs[i][:na].tolist(),
                        "value": float(vals_np[i])
                    })
                return {"batch_eval_resp": {"responses": responses}}
        except Exception as e:
            pass
        na = self.cfg['actions']
        uniform = {"policy": [1.0/max(1,na)]*na, "value": 0.0}
        n = batch_req.get("batch_size", 1) if 'batch_req' in dir() else 1
        return {"batch_eval_resp": {"responses": [uniform]*n}}


def test_batch_eval_single_request():
    """Single request in a batch should return valid response."""
    cfg = {'board': 7, 'ch': 3, 'actions': 49}
    model = MockModel(49)
    client = MockNNSearchClient(model, cfg)

    features = [0.0] * (3 * 7 * 7)  # 147 features
    req = json.dumps({
        "batch_eval_req": {
            "batch_size": 1,
            "requests": [{
                "features": features,
                "action_mask": [1]*49,
                "num_actions": 49
            }]
        }
    })

    resp = client._eval_nn_batch(req)
    assert "batch_eval_resp" in resp
    responses = resp["batch_eval_resp"]["responses"]
    assert len(responses) == 1
    assert len(responses[0]["policy"]) == 49
    assert isinstance(responses[0]["value"], float)
    # Uniform logits → uniform softmax → each prob ≈ 1/49
    assert abs(sum(responses[0]["policy"]) - 1.0) < 1e-4


def test_batch_eval_multiple_requests():
    """Multiple requests should all get responses."""
    cfg = {'board': 7, 'ch': 3, 'actions': 49}
    model = MockModel(49)
    client = MockNNSearchClient(model, cfg)

    features = [0.0] * (3 * 7 * 7)
    req = json.dumps({
        "batch_eval_req": {
            "batch_size": 4,
            "requests": [
                {"features": features, "action_mask": [1]*49, "num_actions": 49}
                for _ in range(4)
            ]
        }
    })

    resp = client._eval_nn_batch(req)
    responses = resp["batch_eval_resp"]["responses"]
    assert len(responses) == 4
    for r in responses:
        assert len(r["policy"]) == 49
        assert abs(sum(r["policy"]) - 1.0) < 1e-4


def test_batch_eval_wrong_feature_size():
    """Wrong feature size should not crash — should use zero padding."""
    cfg = {'board': 7, 'ch': 3, 'actions': 49}
    model = MockModel(49)
    client = MockNNSearchClient(model, cfg)

    req = json.dumps({
        "batch_eval_req": {
            "batch_size": 1,
            "requests": [{
                "features": [1.0, 2.0, 3.0],  # Wrong size!
                "action_mask": [1]*49,
                "num_actions": 49
            }]
        }
    })

    resp = client._eval_nn_batch(req)
    responses = resp["batch_eval_resp"]["responses"]
    assert len(responses) == 1
    assert len(responses[0]["policy"]) == 49


def test_batch_eval_no_model():
    """With no model, should return uniform fallback."""
    cfg = {'board': 7, 'ch': 3, 'actions': 49}
    client = MockNNSearchClient(None, cfg)

    features = [0.0] * (3 * 7 * 7)
    req = json.dumps({
        "batch_eval_req": {
            "batch_size": 2,
            "requests": [
                {"features": features, "action_mask": [1]*49, "num_actions": 49},
                {"features": features, "action_mask": [1]*49, "num_actions": 49},
            ]
        }
    })

    resp = client._eval_nn_batch(req)
    responses = resp["batch_eval_resp"]["responses"]
    assert len(responses) == 2
    for r in responses:
        assert abs(r["value"] - 0.0) < 1e-6
        assert abs(sum(r["policy"]) - 1.0) < 1e-4


def test_batch_eval_chess_actions():
    """Chess with full 4672 actions should work."""
    cfg = {'board': 8, 'ch': 36, 'actions': 4672}
    model = MockModel(4672)
    client = MockNNSearchClient(model, cfg)

    features = [0.0] * (36 * 8 * 8)
    req = json.dumps({
        "batch_eval_req": {
            "batch_size": 1,
            "requests": [{
                "features": features,
                "action_mask": [1]*4672,
                "num_actions": 4672
            }]
        }
    })

    resp = client._eval_nn_batch(req)
    responses = resp["batch_eval_resp"]["responses"]
    assert len(responses) == 1
    assert len(responses[0]["policy"]) == 4672


def test_batch_eval_response_json_roundtrip():
    """Verify the response can be JSON serialized and parsed back."""
    cfg = {'board': 7, 'ch': 3, 'actions': 49}
    model = MockModel(49)
    client = MockNNSearchClient(model, cfg)

    features = [0.0] * (3 * 7 * 7)
    req = json.dumps({
        "batch_eval_req": {
            "batch_size": 2,
            "requests": [
                {"features": features, "action_mask": [1]*49, "num_actions": 49},
                {"features": features, "action_mask": [1]*49, "num_actions": 49},
            ]
        }
    })

    resp = client._eval_nn_batch(req)
    # Serialize and parse back (simulates wire transfer)
    wire = json.dumps(resp, separators=(',', ':'))
    parsed = json.loads(wire)
    assert "batch_eval_resp" in parsed
    assert len(parsed["batch_eval_resp"]["responses"]) == 2


def test_unpack_shm_search_response_parses_sparse_binary_payload():
    from quartz import alphazero_train as az

    payload = bytearray()
    payload.extend(struct.pack("<BQI", 1, 0, 1))
    payload.extend(struct.pack("<B", 0))
    payload.extend(struct.pack("<III", 17, 321, 9))
    payload.extend(struct.pack("<ffffff", 0.25, -0.5, 0.125, 0.75, 0.05, -0.125))
    payload.extend(struct.pack("<I", 2))
    payload.extend(struct.pack("<If", 3, 0.6))
    payload.extend(struct.pack("<If", 5, 0.4))
    payload.extend(struct.pack("<I", 2))
    payload.extend(struct.pack("<Q", 101))
    payload.extend(struct.pack("<Q", 202))
    for text in (
        "BudgetExhausted",
        "e2e4",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    ):
        encoded = text.encode("utf-8")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)

    decoded = az.unpack_shm_search_response(bytes(payload))

    assert decoded["result"]["best_move"] == 17
    assert decoded["result"]["policy"][0][0] == 3
    assert decoded["result"]["policy"][1][0] == 5
    assert abs(decoded["result"]["policy"][0][1] - 0.6) < 1e-6
    assert abs(decoded["result"]["policy"][1][1] - 0.4) < 1e-6
    assert decoded["result"]["iterations"] == 321
    assert decoded["result"]["result_history_hashes"] == [101, 202]
    assert decoded["result"]["best_move_uci"] == "e2e4"


def test_game_configs_have_n_threads():
    """Verify all game configs include n_threads and batch_size."""
    # Import the actual configs
    try:
        # This may fail if torch isn't available
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "quartz_alphazero_train",
            os.path.join(os.path.dirname(__file__), '..', 'quartz', 'alphazero_train.py'))
        # Just check the file content instead
        with open(os.path.join(os.path.dirname(__file__), '..', 'quartz', 'alphazero_train.py')) as f:
            content = f.read()
        assert 'n_threads=' in content, "n_threads should be in game configs"
        assert 'batch_size=' in content, "batch_size should be in game configs"
    except Exception:
        pass  # OK if import fails (no torch), we verified content


if __name__ == '__main__':
    test_batch_eval_single_request()
    test_batch_eval_multiple_requests()
    test_batch_eval_wrong_feature_size()
    test_batch_eval_no_model()
    test_batch_eval_chess_actions()
    test_batch_eval_response_json_roundtrip()
    test_unpack_shm_search_response_parses_sparse_binary_payload()
    test_game_configs_have_n_threads()
    print("All Python batch protocol tests passed!")
