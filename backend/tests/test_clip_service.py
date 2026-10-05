import pytest
import numpy as np
from PIL import Image
from backend.services.clip_service import CLIPService

@pytest.fixture(scope="module")
def clip_service():
    return CLIPService("openai/clip-vit-base-patch32")

@pytest.mark.slow
def test_clip_service_text_embedding_shape_and_norm(clip_service):
    text = "A beautiful sunny day at the beach"
    embedding = clip_service.get_text_embedding(text)
    
    # Assert shape is (1, 512)
    assert embedding.shape == (1, 512)
    
    # Assert L2 norm is approximately 1.0
    norm = np.linalg.norm(embedding[0])
    np.testing.assert_allclose(norm, 1.0, atol=1e-6)

@pytest.mark.slow
def test_clip_service_image_embedding_shape_and_norm(clip_service):
    # Create a dummy image
    image = Image.new("RGB", (224, 224), color="red")
    embedding = clip_service.get_image_embedding(image)
    
    # Assert shape is (1, 512)
    assert embedding.shape == (1, 512)
    
    # Assert L2 norm is approximately 1.0
    norm = np.linalg.norm(embedding[0])
    np.testing.assert_allclose(norm, 1.0, atol=1e-6)
