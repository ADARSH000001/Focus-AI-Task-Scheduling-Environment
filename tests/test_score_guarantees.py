from reward_and_tasks import (
    grade_easy,
    grade_hard,
    grade_medium,
    safe_score,
)


def test_safe_score_stays_in_open_interval():
    for raw in (-5.0, 0.0, 0.25, 0.5, 1.0, 3.0):
        score = safe_score(raw)
        assert 0.0 < score < 1.0


def test_graders_stay_in_open_interval():
    low_metrics = {
        "completed_tasks": 0,
        "total_tasks": 5,
        "on_time": 0,
        "good_energy_usage": 0,
        "total_steps": 5,
        "high_priority_choices": 0,
    }
    high_metrics = {
        "completed_tasks": 5,
        "total_tasks": 5,
        "on_time": 5,
        "good_energy_usage": 5,
        "total_steps": 5,
        "high_priority_choices": 5,
    }
    for grader in (grade_easy, grade_medium, grade_hard):
        assert 0.0 < grader(low_metrics) < 1.0
        assert 0.0 < grader(high_metrics) < 1.0
