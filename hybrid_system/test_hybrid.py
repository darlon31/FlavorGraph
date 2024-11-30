from hybrid_system import HybridRecipeSystem

def print_recipe(recipe):
    """Pretty print a recipe"""
    if not recipe:
        print("Failed to generate recipe")
        return
    
    print("\n=== Generated Recipe ===")
    print(f"\nTitle: {recipe['title']}")
    
    print("\nOriginal Ingredients:")
    for i, ing in enumerate(recipe['metadata']['original_ingredients'], 1):
        print(f"{i}. {ing}")
    
    print("\nEnhanced Ingredients:")
    for i, ing in enumerate(recipe['metadata']['enhanced_ingredients'], 1):
        category = recipe['metadata']['categories'].get(ing, 'Unknown')
        print(f"{i}. {ing} ({category})")
    
    print("\nRecipe Ingredients:")
    for i, ing in enumerate(recipe['ingredients'], 1):
        print(f"{i}. {ing}")
    
    print("\nDirections:")
    for i, step in enumerate(recipe['directions'], 1):
        print(f"{i}. {step}")

def test_recipe_generation():
    """Test the hybrid recipe system"""
    print("Initializing test...")
    
    # Initialize system
    system = HybridRecipeSystem()
    
    # Test cases
    test_cases = [
        ["chicken", "rice", "garlic", "soy sauce"],  # Asian-inspired
        ["pasta", "tomato", "basil", "olive oil"],   # Italian-inspired
        ["beef", "potato", "carrot", "onion"]        # Comfort food
    ]
    
    # Generate and print recipes
    for i, ingredients in enumerate(test_cases, 1):
        print(f"\n\nTest Case {i}: {', '.join(ingredients)}")
        recipe = system.generate_recipe(ingredients)
        print_recipe(recipe)

if __name__ == "__main__":
    test_recipe_generation()
