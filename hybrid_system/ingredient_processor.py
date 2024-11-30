import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class IngredientProcessor:
    def __init__(self):
        """Initialize the processor with FlavorGraph data"""
        print("Loading FlavorGraph data...")
        self.embeddings = self._load_embeddings()
        self.nodes = self._load_nodes()
        self.categories = self._load_categories()
        self.ingredient_to_id = self._create_ingredient_map()
        print("FlavorGraph data loaded successfully!")
    
    def _load_embeddings(self):
        """Load FlavorGraph embeddings"""
        with open(Config.EMBEDDING_FILE, 'rb') as f:
            return pickle.load(f)
    
    def _load_nodes(self):
        """Load node information"""
        return pd.read_csv(Config.NODES_FILE)
    
    def _load_categories(self):
        """Load ingredient categories"""
        return pd.read_csv(Config.CATEGORIES_FILE)
    
    def _create_ingredient_map(self):
        """Create mapping from ingredient names to IDs"""
        return {row['name']: row['id'] for _, row in self.nodes.iterrows()}
    
    def clean_ingredient_name(self, name):
        """Clean and standardize ingredient names"""
        return name.lower().strip().replace(' ', '_')
    
    def find_similar_ingredients(self, ingredient, n=3):
        """Find similar ingredients using embeddings"""
        clean_name = self.clean_ingredient_name(ingredient)
        if clean_name not in self.ingredient_to_id:
            return []
        
        ing_id = self.ingredient_to_id[clean_name]
        ing_embedding = self.embeddings[ing_id]
        
        # Calculate similarities
        similarities = cosine_similarity([ing_embedding], self.embeddings)[0]
        similar_indices = np.argsort(similarities)[-n-1:-1][::-1]  # Exclude self
        
        similar_ingredients = []
        for idx in similar_indices:
            name = self.nodes[self.nodes['id'] == idx]['name'].iloc[0]
            score = similarities[idx]
            if score >= Config.SIMILARITY_THRESHOLD:
                similar_ingredients.append({
                    'name': name,
                    'score': score
                })
        
        return similar_ingredients
    
    def enhance_ingredients(self, ingredients):
        """Enhance ingredient list with compatible ingredients"""
        enhanced = []
        for ing in ingredients:
            enhanced.append(ing)  # Add original ingredient
            similar = self.find_similar_ingredients(
                ing, 
                n=Config.MAX_SIMILAR_INGREDIENTS
            )
            # Add similar ingredients with high compatibility
            for sim in similar:
                enhanced.append(sim['name'])
        
        return list(set(enhanced))  # Remove duplicates
    
    def get_ingredient_category(self, ingredient):
        """Get category for an ingredient"""
        clean_name = self.clean_ingredient_name(ingredient)
        category = self.categories[
            self.categories['ingredient'] == clean_name
        ]['category'].iloc[0] if clean_name in self.categories['ingredient'].values else None
        return category
    
    def process_ingredients(self, ingredients):
        """Main processing pipeline"""
        try:
            # Clean ingredients
            clean_ingredients = [self.clean_ingredient_name(ing) for ing in ingredients]
            
            # Enhance with similar ingredients
            enhanced = self.enhance_ingredients(clean_ingredients)
            
            # Get categories
            categories = {ing: self.get_ingredient_category(ing) for ing in enhanced}
            
            return {
                'original': clean_ingredients,
                'enhanced': enhanced,
                'categories': categories
            }
            
        except Exception as e:
            print(f"Error processing ingredients: {str(e)}")
            return None
