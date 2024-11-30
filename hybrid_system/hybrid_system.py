from ingredient_processor import IngredientProcessor
from recipe_generator import RecipeGenerator

class HybridRecipeSystem:
    def __init__(self):
        """Initialize the hybrid system"""
        print("Initializing Hybrid Recipe System...")
        self.ingredient_processor = IngredientProcessor()
        self.recipe_generator = RecipeGenerator()
        print("Hybrid Recipe System ready!")
    
    def generate_recipe(self, ingredients, enhance=True):
        """Generate a recipe with molecular-aware ingredient enhancement"""
        try:
            # Process ingredients
            processed = self.ingredient_processor.process_ingredients(ingredients)
            if not processed:
                return None
            
            # Use enhanced or original ingredients based on flag
            recipe_ingredients = processed['enhanced'] if enhance else processed['original']
            
            # Generate recipe
            recipe = self.recipe_generator.generate_recipe(recipe_ingredients)
            if not recipe:
                return None
            
            # Add metadata
            recipe['metadata'] = {
                'original_ingredients': processed['original'],
                'enhanced_ingredients': processed['enhanced'],
                'categories': processed['categories']
            }
            
            return recipe
            
        except Exception as e:
            print(f"Error in hybrid system: {str(e)}")
            return None
