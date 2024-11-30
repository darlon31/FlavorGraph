import os
from typing import Dict, Any

class Config:
    """Configuration class for the FlavorGraph Hybrid Recipe System"""
    
    # Base paths
    BASE_DIR: str = os.path.normpath("c:/Users/dario/FlavorGraph")
    FLAVORGRAPH_DIR: str = os.path.normpath(os.path.join(BASE_DIR, "FlavorGraph"))
    CHEFTRANSFORMER_DIR: str = os.path.normpath(os.path.join(BASE_DIR, "chefTransformer"))
    
    # FlavorGraph settings
    EMBEDDING_DIM: int = 300
    MOLECULAR_DIM: int = 881
    
    # Data files
    INPUT_DIR: str = os.path.normpath(os.path.join(FLAVORGRAPH_DIR, "input"))
    OUTPUT_DIR: str = os.path.normpath(os.path.join(FLAVORGRAPH_DIR, "output"))
    NODES_FILE: str = os.path.normpath(os.path.join(INPUT_DIR, "nodes_191120.csv"))
    CATEGORIES_FILE: str = os.path.normpath(os.path.join(INPUT_DIR, "dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv"))
    EMBEDDING_FILE: str = os.path.normpath(os.path.join(OUTPUT_DIR, "kitchenette_embeddings.pkl"))
    
    # Model Configuration
    MODEL_NAME: str = "flax-community/t5-recipe-generation"
    MODEL_CACHE_DIR: str = "model_cache"
    
    # ChefTransformer settings
    GENERATION_CONFIG: Dict[str, Any] = {
        "max_length": 512,
        "min_length": 64,
        "no_repeat_ngram_size": 3,
        "do_sample": True,
        "top_k": 60,
        "top_p": 0.95,
        "temperature": 0.7
    }
    
    # Hybrid system settings
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score for ingredient compatibility
    MAX_SIMILAR_INGREDIENTS: int = 3    # Maximum number of similar ingredients to suggest
    
    @classmethod
    def verify_directories(cls) -> None:
        """Verify all required directories exist"""
        dirs_to_check = [
            cls.BASE_DIR,
            cls.FLAVORGRAPH_DIR,
            cls.CHEFTRANSFORMER_DIR,
            cls.INPUT_DIR,
            cls.OUTPUT_DIR
        ]
        
        for dir_path in dirs_to_check:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    @classmethod
    def verify_files(cls) -> None:
        """Verify all required files exist"""
        files_to_check = [
            cls.NODES_FILE,
            cls.CATEGORIES_FILE,
            cls.EMBEDDING_FILE
        ]
        
        for file_path in files_to_check:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
    
    @classmethod
    def verify_paths(cls) -> None:
        """Verify all required paths and files exist"""
        cls.verify_directories()
        cls.verify_files()