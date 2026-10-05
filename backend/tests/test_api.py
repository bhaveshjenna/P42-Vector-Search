import os
import io
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock the settings before importing main
from unittest import mock

from contextlib import asynccontextmanager

@asynccontextmanager
async def mock_lifespan_context(app):
    yield

import main as main_app
main_app.app.router.lifespan_context = mock_lifespan_context
from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_globals():
    class MockCLIP:
        def get_text_embedding(self, text):
            return np.ones((1, 512), dtype=np.float32)
        def get_image_embedding(self, img):
            return np.ones((1, 512), dtype=np.float32)

    class MockFAISSIndex:
        @property
        def ntotal(self):
            return 100

    class MockFAISS:
        def __init__(self):
            self.index = MockFAISSIndex()
        def search(self, embedding, k=5):
            # Return exact same dist/indices based on k
            return np.array([0.9995, 0.95, 0.85, 0.75, 0.65, 0.55][:k]), np.array([1, 2, 3, 4, 5, 6][:k])

    main_app.clip_service = MockCLIP()
    main_app.image_faiss = MockFAISS()
    main_app.caption_faiss = MockFAISS()
    main_app.mapping = {
        "images": {
            "1": {"filename": "img1.jpg", "captions": ["a dog"]},
            "2": {"filename": "img2.jpg", "captions": ["a cat"]},
            "3": {"filename": "img3.jpg", "captions": []},
            "4": {"filename": "img4.jpg", "captions": []},
            "5": {"filename": "img5.jpg", "captions": []},
            "6": {"filename": "img6.jpg", "captions": []},
        },
        "captions": {
            "1": {"image_id": "1", "text": "a dog"}
        }
    }
    
    # We must also patch os.makedirs because static mount will try to create the mocked IMAGES_DIR
    with mock.patch("os.makedirs"):
        yield

def test_search_text():
    response = client.post("/search/text", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 5
    # verify raw cosine string format (e.g. "0.9995")
    assert data["results"][0]["similarity"] == "0.9995"
    assert data["results"][0]["faiss_id"] == 1

def test_upload_too_large():
    # 10 MB limit
    large_file = b"0" * (10 * 1024 * 1024 + 100)
    response = client.post("/search/image", files={"file": ("large.jpg", large_file, "image/jpeg")})
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()

def test_invalid_image():
    bad_file = b"not an image file"
    response = client.post("/search/image", files={"file": ("bad.jpg", bad_file, "image/jpeg")})
    assert response.status_code == 400
    assert "invalid image" in response.json()["detail"].lower()

def test_exclude_near_duplicates_none_above_threshold():
    from PIL import Image
    import io
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_img = buf.getvalue()

    with mock.patch.object(main_app.image_faiss, "search", return_value=(np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94]), np.array([1, 2, 3, 4, 5, 6]))):
        response = client.post("/search/image?exclude_near_duplicates=true", files={"file": ("test.jpg", valid_img, "image/jpeg")})
        assert response.status_code == 200
        res = response.json()["results"]
        assert len(res) == 5
        assert res[0]["similarity"] == "0.9900"
        assert res[0]["faiss_id"] == 1

def test_exclude_near_duplicates_above_threshold():
    from PIL import Image
    import io
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_img = buf.getvalue()

    with mock.patch.object(main_app.image_faiss, "search", return_value=(np.array([0.9995, 0.95, 0.85, 0.75, 0.65, 0.55]), np.array([1, 2, 3, 4, 5, 6]))):
        # exclude_near_duplicates=false
        response = client.post("/search/image?exclude_near_duplicates=false", files={"file": ("test.jpg", valid_img, "image/jpeg")})
        assert response.status_code == 200
        res = response.json()["results"]
        assert len(res) == 5
        assert res[0]["similarity"] == "0.9995"
        assert res[0]["faiss_id"] == 1

        # exclude_near_duplicates=true
        response = client.post("/search/image?exclude_near_duplicates=true", files={"file": ("test.jpg", valid_img, "image/jpeg")})
        assert response.status_code == 200
        res = response.json()["results"]
        assert len(res) == 5
        assert res[0]["similarity"] == "0.9500"
        assert res[0]["faiss_id"] == 2

def test_health_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["images_indexed"] == 100
    assert data["captions_indexed"] == 100
    assert data["mapping_loaded"] is True

def test_health_unhealthy_missing_index():
    import main as main_app
    old_faiss = main_app.image_faiss
    class MockEmptyFAISS:
        class MockEmptyFAISSIndex:
            @property
            def ntotal(self):
                return 0
        def __init__(self):
            self.index = self.MockEmptyFAISSIndex()
            
    main_app.image_faiss = MockEmptyFAISS()
    try:
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert data["images_indexed"] == 0
        assert data["mapping_loaded"] is True
    finally:
        main_app.image_faiss = old_faiss

def test_empty_query():
    response = client.post("/search/text", json={"query": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

@mock.patch('backend.main.Image.open')
def test_oversized_image_dims_mocked(mock_open):
    class MockImage:
        width = 10000
        height = 9000
        def convert(self, mode): return self
    mock_open.return_value = MockImage()
    
    response = client.post('/search/image', files={'file': ('test.jpg', b'fakebytes', 'image/jpeg')})
    assert response.status_code == 400
    assert 'dimensions are too large' in response.json()['detail'].lower()

