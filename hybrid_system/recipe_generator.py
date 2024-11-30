from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM
from config import Config

class RecipeGenerator:
    def __init__(self):
        """Initialize the ChefTransformer model"""
        print("Loading ChefTransformer model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            use_fast=True
        )
        self.model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            Config.MODEL_NAME
        )
        print("ChefTransformer model loaded successfully!")
    
    def _format_ingredients(self, ingredients):
        """Format ingredients for model input"""
        return ", ".join(ingredients)
    
    def _parse_recipe(self, recipe_text):
        """Parse generated recipe text into structured format"""
        sections = recipe_text.split("\n")
        recipe = {
            'title': '',
            'ingredients': [],
            'directions': []
        }
        
        current_section = None
        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                recipe['title'] = section.replace("title:", "").strip()
                current_section = 'title'
            elif section.startswith("ingredients:"):
                current_section = 'ingredients'
                items = section.replace("ingredients:", "").split("--")
                recipe['ingredients'].extend([i.strip() for i in items if i.strip()])
            elif section.startswith("directions:"):
                current_section = 'directions'
                steps = section.replace("directions:", "").split("--")
                recipe['directions'].extend([s.strip() for s in steps if s.strip()])
            elif section and current_section:
                if current_section == 'ingredients':
                    recipe['ingredients'].append(section)
                elif current_section == 'directions':
                    recipe['directions'].append(section)
        
        return recipe
    
    def generate_recipe(self, ingredients):
        """Generate a recipe from ingredients"""
        try:
            # Format ingredients
            input_text = f"items: {self._format_ingredients(ingredients)}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="jax"
            )
            
            # Generate recipe
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **Config.GENERATION_CONFIG
            )
            
            # Decode and parse recipe
            recipe_text = self.tokenizer.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
            return self._parse_recipe(recipe_text)
            
        except Exception as e:
            print(f"Error generating recipe: {str(e)}")
            return None
