import importlib
import sys
import types
import pytest


class DummyDB:
    def people(self):
        return []


class DummyPipeline:
    """Lightweight stand-in for app.core.recognize.Pipeline."""
    def __init__(self):
        self.db = DummyDB()

    def enroll_image(self, img, person_id: str, source: str):
        return {"faces": 1, "embeddings_added": 1}

    def recognize_image(self, img):
        return None


@pytest.fixture
def server_module(monkeypatch):
    """
    Import app.api.server with Pipeline stubbed BEFORE import,
    and force a clean import every time.
    """
    # 1) Stub the module path: app.core.recognize.Pipeline
    recognize_mod = types.SimpleNamespace(Pipeline=DummyPipeline)
    monkeypatch.setitem(sys.modules, "app.core.recognize", recognize_mod)

    # 2) Ensure server is not cached
    sys.modules.pop("app.api.server", None)

    import app.api.server as server
    importlib.reload(server)  # extra safety

    return server


@pytest.fixture
def config_module():
    import app.config as config
    return config
