"""Tests for the 1990s Indian Nostalgia direct AI-video pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

from autocast.config import Config
from autocast.orchestrator import run_pipeline
from autocast.providers.video import generate_scene_video
from autocast.spine import Character, CharacterBible, Run, Scene, StageStatus
from autocast.stages import direction, images, script, topic, video


def _cfg(tmp_path: Path) -> Config:
    return Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        gemini_api_key="test-gemini-key-never-exposed",
    )


def test_character_bible_and_scene_spine_roundtrip(tmp_path):
    run = Run.new("2026-07-21")
    char = Character(
        name="Aarav",
        age="9 years old",
        gender="male",
        height="4'2\"",
        body_type="lean",
        skin_tone="wheatish",
        face_shape="round",
        eyes="brown",
        hair="black",
        hairstyle="short crop",
        clothing="yellow t-shirt",
        footwear="chappals",
        accessories="black wrist thread",
        unique_facial_features="beauty mark on cheek",
        personality="curious",
        typical_expressions="wonder",
        body_language="energetic",
    )
    run.character_bible = CharacterBible(characters=[char])
    run.scenes = [
        Scene(
            scene_number=1,
            title_hindi="नानी का आँगन",
            characters_hindi="आरव और नानी",
            location_hindi="गाँव का आँगन",
            action_hindi="आँगन में खेलना",
            dialogue_hindi="नानी, आम का पन्ना तैयार है क्या?",
            video_prompt="1990s Ghibli-inspired hand-drawn 2D animation...",
            duration_s=6.5,
            clip_path="assets/scene_001.mp4",
            has_native_audio=True,
            status="completed",
        )
    ]
    p = tmp_path / "run.json"
    run.save(p)
    loaded = Run.load(p)

    assert loaded.character_bible is not None
    assert len(loaded.character_bible.characters) == 1
    assert loaded.character_bible.characters[0].name == "Aarav"
    assert len(loaded.scenes) == 1
    assert loaded.scenes[0].title_hindi == "नानी का आँगन"
    assert loaded.scenes[0].status == "completed"


def test_direction_generates_character_bible_and_scenes(tmp_path):
    cfg = _cfg(tmp_path)
    spine = Run.new("2026-07-21")
    spine = topic.run(spine, cfg, dry_run=True)
    spine = script.run(spine, cfg, dry_run=True)
    spine = direction.run(spine, cfg, dry_run=True)

    assert spine.character_bible is not None
    assert len(spine.character_bible.characters) >= 2
    assert len(spine.scenes) >= 3
    assert all(s.video_prompt for s in spine.scenes)
    assert all("1990s Ghibli-inspired" in s.video_prompt for s in spine.scenes)
    assert all("Negative prompt:" in s.video_prompt for s in spine.scenes)
    assert all(s.has_native_audio for s in spine.scenes)


def test_dry_video_generation_provider(tmp_path):
    cfg = _cfg(tmp_path)
    out_mp4 = str(tmp_path / "scene_001.mp4")
    out_png = str(tmp_path / "scene_001.png")

    result = generate_scene_video(
        cfg,
        prompt="1990s Ghibli-inspired animation of Indian village",
        out_mp4=out_mp4,
        out_png=out_png,
        duration_s=5.0,
        dry_run=True,
    )
    assert Path(result).exists()
    assert Path(out_png).exists()


def test_per_scene_resumability(tmp_path):
    """If scene 1 is already completed, re-running images stage must skip scene 1."""
    cfg = _cfg(tmp_path)
    assets_dir = cfg.assets_dir("2026-07-21")
    assets_dir.mkdir(parents=True, exist_ok=True)

    spine = Run.new("2026-07-21")
    spine = topic.run(spine, cfg, dry_run=True)
    spine = script.run(spine, cfg, dry_run=True)
    spine = direction.run(spine, cfg, dry_run=True)

    # Pre-complete scene 1
    scene1_file = assets_dir / "scene_001.mp4"
    scene1_file.write_bytes(b"existing-video-content-scene-1")
    spine.scenes[0].status = "completed"
    spine.scenes[0].clip_path = "assets/scene_001.mp4"

    spine = images.run(spine, cfg, dry_run=True)

    # Scene 1 content was preserved and not overwritten
    assert scene1_file.read_bytes() == b"existing-video-content-scene-1"
    # All scenes are now completed
    assert all(s.status == "completed" for s in spine.scenes)


def test_end_to_end_nostalgia_dry_run_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    spine = run_pipeline("2026-07-21", cfg, dry_run=True)

    # Pipeline completed
    assert all(rec.status is StageStatus.COMPLETED for rec in spine.stages)
    assert spine.character_bible is not None
    assert len(spine.character_bible.characters) >= 2
    assert len(spine.scenes) >= 3

    # Output files exist
    assets_dir = cfg.assets_dir("2026-07-21")
    assert (assets_dir / "scene_001.mp4").exists()
    assert (assets_dir / "reel.mp4").exists()
    assert (assets_dir / "final.mp4").exists()
    assert (assets_dir / "thumb.jpg").exists()


def test_env_is_gitignored():
    """Verify that .env is ignored by Git and never tracked."""
    proc = subprocess.run(
        ["git", "status", "--ignored"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ".env" in proc.stdout or "nothing to commit" in proc.stdout
    # Also verify git check-ignore directly
    proc2 = subprocess.run(
        ["git", "check-ignore", ".env"],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0


def test_anti_text_prompt_in_direction_scenes(tmp_path):
    """Verify that every generated scene prompt strictly includes anti-text instructions."""
    cfg = _cfg(tmp_path)
    spine = Run.new("2026-07-21")
    spine = topic.run(spine, cfg, dry_run=True)
    spine = script.run(spine, cfg, dry_run=True)
    spine = direction.run(spine, cfg, dry_run=True)

    for scene in spine.scenes:
        prompt = scene.video_prompt
        assert "Spoken dialogue must exist ONLY as native audio" in prompt
        assert "No subtitles, captions" in prompt
        assert "Do not visually render the dialogue as text" in prompt


def test_video_provider_cascade_ordering(tmp_path):
    """Verify that build_video_providers places Agnes first by default, followed by Veo."""
    from autocast.providers.video import build_video_providers

    # 1. Both keys present: Agnes is primary
    cfg_both = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        agnes_api_key="mock-agnes-key",
        gemini_api_key="mock-gemini-key",
        video_provider="agnes",
    )
    providers = build_video_providers(
        cfg_both,
        prompt="test prompt",
        out_mp4=str(tmp_path / "scene.mp4"),
        dry_run=False,
    )
    names = [p.name for p in providers]
    assert names[0] == "agnes-video"
    assert names[1] == "gemini-veo"

    # 2. Preferred Veo: Veo is primary
    cfg_veo = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        agnes_api_key="mock-agnes-key",
        gemini_api_key="mock-gemini-key",
        video_provider="veo",
    )
    providers_veo = build_video_providers(
        cfg_veo,
        prompt="test prompt",
        out_mp4=str(tmp_path / "scene.mp4"),
        dry_run=False,
    )
    names_veo = [p.name for p in providers_veo]
    assert names_veo[0] == "gemini-veo"
    assert names_veo[1] == "agnes-video"

    # 3. Dry run: dry-stub only
    providers_dry = build_video_providers(
        cfg_both,
        prompt="test prompt",
        out_mp4=str(tmp_path / "scene.mp4"),
        dry_run=True,
    )
    assert len(providers_dry) == 1
    assert providers_dry[0].name == "dry-stub"


def test_agnes_provider_mocked_success(tmp_path, monkeypatch):
    """Test Agnes provider request construction, polling, and MP4 download with mocked HTTP responses."""
    import httpx
    from autocast.providers.video import generate_agnes_video

    cfg = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        agnes_api_key="mock-agnes-secret-key",
        agnes_model="agnes-video-v2.0",
        agnes_api_base="https://apihub.agnes-ai.com",
        agnes_rate_limit_seconds=0.0,
    )

    out_mp4 = tmp_path / "mock_scene.mp4"
    out_png = tmp_path / "mock_scene.png"

    calls = []

    def mock_post(url, headers, json):
        calls.append(("POST", url, headers, json))
        assert headers.get("Authorization") == "Bearer mock-agnes-secret-key"
        assert json.get("model") == "agnes-video-v2.0"
        assert "Spoken dialogue must exist ONLY as native audio" in json.get("prompt")
        resp = httpx.Response(200, json={"video_id": "test_video_123"})
        return resp

    poll_count = 0

    def mock_get(url, headers=None):
        nonlocal poll_count
        calls.append(("GET", url))
        if "agnesapi" in url or "videos/test_video_123" in url:
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(200, json={"status": "in_progress"})
            return httpx.Response(
                200,
                json={"status": "completed", "url": "https://cdn.example.com/test_video_123.mp4"},
            )
        return httpx.Response(404)

    class MockStreamContext:
        def __init__(self, method, url):
            self.method = method
            self.url = url

        def __enter__(self):
            class MockStreamResp:
                status_code = 200

                def iter_bytes(self, chunk_size=65536):
                    yield b"MOCK_MP4_BINARY_DATA"

            return MockStreamResp()

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, headers=None, json=None: mock_post(url, headers, json))
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, headers=None: mock_get(url, headers))
    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url: MockStreamContext(method, url))

    result_path = generate_agnes_video(
        cfg,
        prompt="A boy smiling in village",
        out_mp4=str(out_mp4),
        out_png=str(out_png),
        duration_s=5.0,
        dry_run=False,
        poll_interval_s=0,
    )

    assert Path(result_path).exists()
    assert Path(result_path).read_bytes() == b"MOCK_MP4_BINARY_DATA"
    assert any(c[0] == "POST" and "v1/videos" in c[1] for c in calls)
    assert any(c[0] == "GET" and "agnesapi" in c[1] for c in calls)


def test_agnes_provider_mocked_failure(tmp_path, monkeypatch):
    """Test Agnes provider handles server-reported failure cleanly."""
    import httpx
    import pytest
    from autocast.providers.video import generate_agnes_video

    cfg = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        agnes_api_key="mock-agnes-key",
        agnes_rate_limit_seconds=0.0,
    )

    out_mp4 = tmp_path / "fail_scene.mp4"

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, headers=None, json=None: httpx.Response(200, json={"video_id": "fail_123"}),
    )
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, headers=None: httpx.Response(
            200, json={"status": "failed", "error": "Internal GPU error during rendering"}
        ),
    )

    with pytest.raises(RuntimeError, match="generation failed on server"):
        generate_agnes_video(
            cfg,
            prompt="Prompt that fails",
            out_mp4=str(out_mp4),
            duration_s=5.0,
            dry_run=False,
            poll_interval_s=0,
        )


def test_agnes_rate_limiter_standalone_spacing():
    """Verify that AgnesRateLimiter enforces 60s minimum between consecutive requests."""
    from autocast.providers.video import AgnesRateLimiter

    limiter = AgnesRateLimiter(min_interval_s=60.0)

    # 1. First request has no previous timestamp -> 0 wait
    assert limiter.wait_if_needed() == 0.0
    limiter.record_request()

    # 2. Immediate second request (0s elapsed) -> waits 60s
    slept = []
    wait_time = limiter.wait_if_needed(sleep_fn=lambda s: slept.append(s))
    assert wait_time > 59.0
    assert len(slept) == 1 and slept[0] > 59.0

    # 3. Request after 65s has elapsed -> 0 wait
    limiter._last_request_time = limiter._last_request_time - 65.0  # simulate time passing
    slept.clear()
    wait_time_after = limiter.wait_if_needed(sleep_fn=lambda s: slept.append(s))
    assert wait_time_after == 0.0
    assert len(slept) == 0


def test_agnes_rate_limiter_integrated_into_generator(tmp_path, monkeypatch):
    """Verify consecutive generate_agnes_video calls are never submitted <60s apart."""
    import httpx
    from autocast.providers.video import AGNES_LIMITER, generate_agnes_video

    AGNES_LIMITER.reset()

    cfg = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        agnes_api_key="mock-agnes-key",
    )

    out1 = tmp_path / "s1.mp4"
    out2 = tmp_path / "s2.mp4"

    simulated_now = 1000.0

    def mock_monotonic():
        return simulated_now

    sleeps = []

    def mock_sleep(seconds):
        nonlocal simulated_now
        sleeps.append(seconds)
        simulated_now += seconds

    monkeypatch.setattr("time.monotonic", mock_monotonic)
    monkeypatch.setattr("time.sleep", mock_sleep)

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda self, url, headers=None, json=None: httpx.Response(200, json={"video_id": "v_1"}),
    )
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, headers=None: httpx.Response(200, json={"status": "completed", "url": "https://cdn.example.com/v.mp4"}),
    )

    class MockStreamContext:
        def __enter__(self):
            class Resp:
                status_code = 200
                def iter_bytes(self, chunk_size=65536):
                    yield b"MP4"
            return Resp()
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url: MockStreamContext())

    # Scene 1: Generation submitted at t=1000.0
    generate_agnes_video(cfg, prompt="scene 1", out_mp4=str(out1), poll_interval_s=0)

    # Simulate scene 1 completed in 20 seconds -> current time is 1020.0
    simulated_now = 1020.0

    # Scene 2: Next generation submitted -> must wait remaining 40 seconds (until 1060.0)
    generate_agnes_video(cfg, prompt="scene 2", out_mp4=str(out2), poll_interval_s=0)

    # Verify that sleep was called for 40.0 seconds before Scene 2 was submitted
    assert any(abs(s - 40.0) < 0.1 for s in sleeps)


def test_agnes_provider_429_backoff_and_retry(tmp_path, monkeypatch):
    """Verify HTTP 429 triggers bounded exponential backoff and retries."""
    import httpx
    from autocast.providers.video import AGNES_LIMITER, generate_agnes_video

    AGNES_LIMITER.reset()

    cfg = Config(
        runs_dir=tmp_path / "runs",
        queue_path=tmp_path / "queue" / "topics.json",
        agnes_api_key="mock-agnes-key",
    )

    attempts = 0
    slept = []

    def mock_post(url, headers, json):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "Rate limit exceeded"})
        return httpx.Response(200, json={"video_id": "v_success"})

    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, headers=None, json=None: mock_post(url, headers, json))
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, headers=None: httpx.Response(200, json={"status": "completed", "url": "https://cdn.example.com/v.mp4"}),
    )

    class MockStreamContext:
        def __enter__(self):
            class Resp:
                status_code = 200
                def iter_bytes(self, chunk_size=65536):
                    yield b"MP4"
            return Resp()
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url: MockStreamContext())

    res = generate_agnes_video(cfg, prompt="test 429", out_mp4=str(tmp_path / "out.mp4"), poll_interval_s=0)
    assert Path(res).exists()
    assert attempts == 2
    assert len(slept) >= 1
    assert slept[0] == 60.0


