from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from config import Config

class RecipeGenerator:
    def __init__(self):
        """Initialize the ChefTransformer model"""
        print("Loading ChefTransformer model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODEL_CACHE_DIR,
            local_files_only=False
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODEL_CACHE_DIR,
            local_files_only=False
        ).to(self.device)
        print(f"ChefTransformer model loaded successfully on {self.device}!")
    
    def _format_ingredients(self, ingredients):
        """Format ingredients for model input"""
        return f"ingredients: {', '.join(ingredients)}"
    
    def _parse_recipe(self, recipe_text):
        """Parse generated recipe text into structured format"""
        recipe = {
            'title': '',
            'ingredients': [],
            'directions': []
        }
        
        # Split into sections
        sections = recipe_text.lower().split("ingredients:")
        if len(sections) >= 2:
            # Get title
            title_part = sections[0].strip()
            if "title:" in title_part:
                recipe['title'] = title_part.replace("title:", "").strip()
            
            # Process ingredients and directions
            rest = sections[1].split("directions:")
            if len(rest) >= 2:
                # Get ingredients
                ingredients_text = rest[0].strip()
                recipe['ingredients'] = [
                    ing.strip() 
                    for ing in ingredients_text.split(",") 
                    if ing.strip()
                ]
                
                # Get directions
                directions_text = rest[1].strip()
                recipe['directions'] = [
                    step.strip() 
                    for step in directions_text.split(".") 
                    if step.strip() and step.strip().isdigit() == False
                ]
        
        return recipe
    
    def generate_recipe(self, ingredients, max_length=512, temperature=0.7):
        """Generate a recipe from a list of ingredients"""
        try:
            # Format input
            input_text = self._format_ingredients(ingredients)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate recipe
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=64,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=temperature,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and parse recipe
            recipe_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_recipe(recipe_text)
            
        except Exception as e:
            print(f"Error generating recipe: {str(e)}")
            return None
