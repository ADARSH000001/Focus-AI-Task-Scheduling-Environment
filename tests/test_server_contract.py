from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_health_and_validate():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    validate = client.get("/validate")
    assert validate.status_code == 200
    assert validate.json()["valid"] is True


def test_reset_and_step():
    reset = client.post("/reset", json={"difficulty": "easy"})
    assert reset.status_code == 200
    body = reset.json()
    assert "observation" in body
    assert "observation_text" in body
    first_task_id = body["observation"]["tasks"][0]["id"]

    step = client.post("/step", json={"action": f"start_task('{first_task_id}')"})
    assert step.status_code == 200
    step_body = step.json()
    assert "reward" in step_body
    assert "info" in step_body
