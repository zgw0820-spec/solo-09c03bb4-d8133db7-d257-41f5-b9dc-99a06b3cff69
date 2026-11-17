"""API contract tests against the real FastAPI app and a real SQLite file."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(app) as test_client:
        yield test_client


def payload(records: int = 8, epochs: int = 2):
    rng = np.random.default_rng(1)
    return {
        "features": rng.normal(size=(records, 2)).tolist(),
        "labels": [i % 2 for i in range(records)],
        "epochs": epochs,
    }


def test_swagger_and_health(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/jobs" in schema["paths"]
    assert "/jobs/{job_id}/checkpoints" in schema["paths"]
    health = client.get("/health").json()
    assert health["status"] == "ok"


def test_submit_returns_job_metadata(client):
    response = client.post("/jobs", json=payload(12, 3))
    assert response.status_code == 201
    job = response.json()
    assert job["batch_size"] == 8
    assert job["learning_rate"] == 0.1
    assert job["momentum"] == 0.9
    assert job["batches_per_epoch"] == 2
    assert job["total_steps"] == 6
    assert job["completed_steps"] == 0
    assert job["status"] == "queued"
    assert len(job["weights"]) == 3  # two standardised columns + bias


def test_query_progress_and_checkpoints(client):
    job = client.post("/jobs", json=payload(16, 2)).json()
    settings = Settings.from_env()

    from app import lease as lm
    from app import store

    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, job["job_id"], "w1", settings.lease_seconds)
    for _ in range(2):
        store.advance_job(settings.db_path, lease)

    current = client.get(f"/jobs/{job['job_id']}").json()
    assert current["completed_steps"] == 2
    assert current["total_steps"] == 4
    assert current["progress"] == pytest.approx(0.5)
    assert current["latest_checkpoint_step"] == 2
    assert current["final_weights"] is None

    checkpoints = client.get(f"/jobs/{job['job_id']}/checkpoints").json()
    assert [c["step"] for c in checkpoints] == [1, 2]
    assert checkpoints[0]["batch_start_index"] == 0
    assert checkpoints[1]["batch_start_index"] == 8
    assert all(c["worker_id"] == "w1" for c in checkpoints)
    assert all(len(c["weights"]) == 3 and len(c["momentum_velocity"]) == 3 for c in checkpoints)
    assert checkpoints[0]["training_epoch"] == 1 and checkpoints[0]["batch_in_epoch"] == 1
    assert checkpoints[1]["training_epoch"] == 1 and checkpoints[1]["batch_in_epoch"] == 2

    after = client.get(f"/jobs/{job['job_id']}/checkpoints", params={"after_step": 1}).json()
    assert [c["step"] for c in after] == [2]

    events = client.get(f"/jobs/{job['job_id']}/events").json()
    assert events[0]["event"] == "job_created"
    assert events[0]["detail"]["total_steps"] == 4

    workers = client.get("/workers").json()
    assert [w["worker_id"] for w in workers["workers"]] == ["w1"]
    assert workers["workers"][0]["alive"] is True
    assert workers["leases"][0]["lease_epoch"] == 1
    assert workers["leases"][0]["worker_id"] == "w1"


def test_validation_errors(client):
    bad = payload(7)
    assert client.post("/jobs", json=bad).status_code == 422
    assert client.post("/jobs", json=payload(8, 1)).status_code == 422
    assert client.post("/jobs", json=payload(8, 6)).status_code == 422
    mismatch = payload(8)
    mismatch["labels"] = mismatch["labels"][:-1]
    assert client.post("/jobs", json=mismatch).status_code == 422
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/checkpoints").status_code == 404


def test_workers_endpoint_empty(client):
    data = client.get("/workers").json()
    assert "lease_seconds" in data
    assert data["workers"] == []
    assert data["leases"] == []


def test_cancel_endpoint_in_swagger(client):
    schema = client.get("/openapi.json").json()
    assert "/jobs/{job_id}/cancel" in schema["paths"]
    assert "post" in schema["paths"]["/jobs/{job_id}/cancel"]


def test_cancel_queued_job_freezes_it_and_is_idempotent(client):
    job = client.post("/jobs", json=payload(12, 3)).json()
    response = client.post(
        f"/jobs/{job['job_id']}/cancel", json={"reason": "wrong labels"}
    )
    assert response.status_code == 200
    cancelled = response.json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["already_cancelled"] is False
    assert cancelled["cancel_reason"] == "wrong labels"
    assert cancelled["completed_steps"] == 0
    assert cancelled["progress"] == 0.0
    assert cancelled["final_weights"] is None
    assert cancelled["cancelled_at"] == cancelled["finished_at"]

    # Repeated cancellation keeps the original terminal result.
    again = client.post(
        f"/jobs/{job['job_id']}/cancel", json={"reason": "different reason"}
    ).json()
    assert again["already_cancelled"] is True
    assert again["status"] == "cancelled"
    assert again["cancel_reason"] == "wrong labels"
    assert again["finished_at"] == cancelled["finished_at"]
    assert again["updated_at"] == cancelled["updated_at"]

    current = client.get(f"/jobs/{job['job_id']}").json()
    assert current["status"] == "cancelled"
    assert current["cancel_reason"] == "wrong labels"
    assert client.get(f"/jobs/{job['job_id']}/checkpoints").json() == []
    events = client.get(f"/jobs/{job['job_id']}/events").json()
    cancel_events = [e for e in events if e["event"] == "job_cancelled"]
    assert len(cancel_events) == 1
    assert cancel_events[0]["detail"]["reason"] == "wrong labels"
    assert cancel_events[0]["detail"]["completed_steps"] == 0


def test_cancel_without_body_uses_none_reason(client):
    job = client.post("/jobs", json=payload(8, 2)).json()
    response = client.post(f"/jobs/{job['job_id']}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["cancel_reason"] is None


def test_cancel_running_job_keeps_last_checkpoint_and_releases_lease(client):
    from app import lease as lm
    from app import store

    job = client.post("/jobs", json=payload(16, 2)).json()
    settings = Settings.from_env()
    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, job["job_id"], "w1", settings.lease_seconds)
    store.advance_job(settings.db_path, lease)
    snapshot = client.get(f"/jobs/{job['job_id']}").json()

    response = client.post(f"/jobs/{job['job_id']}/cancel", json={"reason": "bad"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "cancelled"
    assert result["completed_steps"] == snapshot["completed_steps"] == 1
    assert result["weights"] == snapshot["weights"]
    assert result["loss"] == snapshot["loss"]
    assert result["momentum_velocity"] == snapshot["momentum_velocity"]

    # The confirmed checkpoint is retained for investigation.
    checkpoints = client.get(f"/jobs/{job['job_id']}/checkpoints").json()
    assert [c["step"] for c in checkpoints] == [1]

    # The lease row is gone; workers must not show this job as leased.
    workers = client.get("/workers").json()
    assert workers["leases"] == []


def test_cancel_completed_job_is_refused(client):
    from app import lease as lm
    from app import store

    job = client.post("/jobs", json=payload(8, 2)).json()
    settings = Settings.from_env()
    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, job["job_id"], "w1", settings.lease_seconds)
    while store.advance_job(settings.db_path, lease)["status"] != "completed":
        pass

    response = client.post(f"/jobs/{job['job_id']}/cancel", json={"reason": "late"})
    assert response.status_code == 409
    current = client.get(f"/jobs/{job['job_id']}").json()
    assert current["status"] == "completed"
    assert current["final_weights"] is not None
    events = client.get(f"/jobs/{job['job_id']}/events").json()
    assert all(e["event"] != "job_cancelled" for e in events)


def test_cancel_unknown_job_returns_404(client):
    response = client.post("/jobs/nope/cancel", json={"reason": "x"})
    assert response.status_code == 404


def test_cancelled_job_is_not_claimable(client):
    from app import store

    settings = Settings.from_env()
    job = client.post("/jobs", json=payload(8, 2)).json()
    client.post(f"/jobs/{job['job_id']}/cancel", json={"reason": "x"})
    assert store.claimable_jobs(settings.db_path) == []
