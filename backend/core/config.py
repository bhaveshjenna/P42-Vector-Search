import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    
    @property
    def IMAGES_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "images")
        
    @property
    def INDEX_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "index")
        
    @property
    def IMAGES_INDEX_FILE(self) -> str:
        return os.path.join(self.INDEX_DIR, "images.index")
        
    @property
    def CAPTIONS_INDEX_FILE(self) -> str:
        return os.path.join(self.INDEX_DIR, "captions.index")
        
    @property
    def MAPPING_FILE(self) -> str:
        return os.path.join(self.INDEX_DIR, "mapping.json")
        
    @property
    def CAPTIONS_FILE(self) -> str:
        return os.path.join(self.DATA_DIR, "captions.txt")
    
    MODEL_ID: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM: int = 512
    
    class Config:
        env_file = ".env"

settings = Settings()
