from scripts.build_features import DEFAULT_SESSION_CONFIG, load_session_config


def test_build_features_defaults_to_versioned_session_config():
    cfg = load_session_config(None)

    assert DEFAULT_SESSION_CONFIG.name == "research_sessions_v1.json"
    assert cfg.broker_timezone == "EET"
    assert cfg.asia_timezone == "Asia/Tokyo"
    assert cfg.asia_start == "09:00"
    assert cfg.asia_end == "18:00"
    assert cfg.london_timezone == "Europe/London"
    assert cfg.london_start == "08:00"
    assert cfg.london_end == "17:00"
    assert cfg.new_york_timezone == "America/New_York"
    assert cfg.new_york_start == "08:00"
    assert cfg.new_york_end == "17:00"
