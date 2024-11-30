import os
from config import Config

def test_config():
    """Test the configuration settings and paths"""
    print("\n=== Testing Configuration ===\n")
    
    # Test 1: Check if paths are correctly normalized
    print("Test 1: Path Normalization")
    paths_to_check = [
        ("BASE_DIR", Config.BASE_DIR),
        ("FLAVORGRAPH_DIR", Config.FLAVORGRAPH_DIR),
        ("INPUT_DIR", Config.INPUT_DIR),
        ("OUTPUT_DIR", Config.OUTPUT_DIR)
    ]
    
    for name, path in paths_to_check:
        normalized = os.path.normpath(path)
        if path == normalized:
            print(f"[PASS] {name} is properly normalized: {path}")
        else:
            print(f"[FAIL] {name} is not normalized. Found: {path}, Expected: {normalized}")
    
    # Test 2: Verify directories exist
    print("\nTest 2: Directory Verification")
    try:
        Config.verify_directories()
        print("[PASS] All required directories exist!")
    except FileNotFoundError as e:
        print(f"[FAIL] Error: {str(e)}")
    
    # Test 3: Verify files exist
    print("\nTest 3: File Verification")
    try:
        Config.verify_files()
        print("[PASS] All required files exist!")
    except FileNotFoundError as e:
        print(f"[FAIL] Error: {str(e)}")
    
    # Test 4: Check generation config
    print("\nTest 4: Generation Config")
    required_keys = {
        "max_length": int,
        "min_length": int,
        "do_sample": bool,
        "top_k": int,
        "top_p": float,
        "temperature": float
    }
    
    for key, expected_type in required_keys.items():
        if key not in Config.GENERATION_CONFIG:
            print(f"[FAIL] Missing required key: {key}")
            continue
            
        value = Config.GENERATION_CONFIG[key]
        if not isinstance(value, expected_type):
            print(f"[FAIL] Wrong type for {key}. Expected {expected_type}, got {type(value)}")
        else:
            print(f"[PASS] {key} has correct type: {expected_type}")
    
    # Test 5: Check hybrid system settings
    print("\nTest 5: Hybrid System Settings")
    if not (0 <= Config.SIMILARITY_THRESHOLD <= 1):
        print(f"[FAIL] SIMILARITY_THRESHOLD should be between 0 and 1, got {Config.SIMILARITY_THRESHOLD}")
    else:
        print(f"[PASS] SIMILARITY_THRESHOLD is valid: {Config.SIMILARITY_THRESHOLD}")
    
    if Config.MAX_SIMILAR_INGREDIENTS <= 0:
        print(f"[FAIL] MAX_SIMILAR_INGREDIENTS should be positive, got {Config.MAX_SIMILAR_INGREDIENTS}")
    else:
        print(f"[PASS] MAX_SIMILAR_INGREDIENTS is valid: {Config.MAX_SIMILAR_INGREDIENTS}")

if __name__ == "__main__":
    test_config()
