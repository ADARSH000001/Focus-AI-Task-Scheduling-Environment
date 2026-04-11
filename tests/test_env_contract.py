from env import FocusEnv, smart_agent
from models import Observation, Reward


def test_reset_and_step_contract():
    env = FocusEnv(difficulty="easy")
    obs = env.reset(seed=0)
    assert isinstance(obs, Observation)
    assert obs.legal_actions

    obs, reward, done, info = env.step(obs.legal_actions[0])
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert "metrics" in info


def test_invalid_actions_trigger_stagnation_done():
    env = FocusEnv(difficulty="easy")
    env.reset(seed=0)
    done = False
    info = {}
    for _ in range(5):
        _, reward, done, info = env.step("not_a_real_action")
        assert reward.reward == -3.0
    assert done is True
    assert 0.0 < info["score"] < 1.0


def test_seeded_episode_score_range():
    for difficulty in ("easy", "medium", "hard"):
        env = FocusEnv(difficulty=difficulty)
        obs = env.reset(seed=7)
        score = None
        for _ in range(20):
            obs, _, done, info = env.step(smart_agent(obs))
            if done:
                score = info["score"]
                break
        assert score is not None
        assert 0.0 < score < 1.0
