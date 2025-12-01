# train_ollama_model.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import ollama
import time
from datetime import datetime
import logging

class OllamaInterviewTrainer:
    def __init__(self, database_path='database.jsonl', model_name="interview-expert"):
        self.database_path = database_path
        self.model_name = model_name
        self.data = None
        self.vectorizer = None
        self.classifier = None
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the training data"""
        self.logger.info("Loading training data...")
        
        try:
            # Read the JSONL file
            data = []
            with open(self.database_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping invalid JSON line: {e}")
            
            self.data = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(self.data)} records from database")
            
            # Basic preprocessing
            if 'category' not in self.data.columns:
                self.data['category'] = 'general'
            
            # Ensure all required columns exist
            required_columns = ['question', 'category', 'ideal_answer']
            for col in required_columns:
                if col not in self.data.columns:
                    self.data[col] = ''
            
            # Clean data
            self.data = self.data.dropna(subset=['question'])
            self.data['question'] = self.data['question'].str.strip()
            self.data['category'] = self.data['category'].str.lower()
            
            self.logger.info(f"Final dataset size: {len(self.data)} records")
            self.logger.info(f"Categories: {self.data['category'].value_counts().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_training_prompts(self):
        """Prepare training prompts for Ollama fine-tuning"""
        self.logger.info("Preparing training prompts...")
        
        training_data = []
        
        for _, row in self.data.iterrows():
            # Create training prompt in chat format
            prompt = {
                "question": row['question'],
                "category": row.get('category', 'general'),
                "ideal_answer": row.get('ideal_answer', ''),
                "expected_keywords": row.get('expected_keywords', []),
                "difficulty": row.get('difficulty', 'medium')
            }
            training_data.append(prompt)
        
        return training_data
    
    def create_modelfile(self, training_data):
        """Create Ollama Modelfile for training"""
        self.logger.info("Creating Modelfile...")
        
        modelfile_content = f"""FROM llama2

# System prompt for interview expert
SYSTEM \"\"\"You are an AI Interview Expert specializing in technical and behavioral interviews.
Your role is to:
1. Generate relevant interview questions based on categories
2. Evaluate answers comprehensively
3. Provide constructive feedback
4. Suggest improvements
5. Maintain professional and helpful tone

You have been trained on {len(training_data)} interview questions and answers.
Always respond in a structured, professional manner.\"\"\"

# Training data - converting to conversation format
"""
        
        # Add training examples
        for i, example in enumerate(training_data[:1000]):  # Limit to first 1000 for modelfile
            modelfile_content += f"""
# Example {i+1}
MESSAGE user {{
    "role": "user",
    "content": "Generate a {example['category']} interview question with difficulty {example['difficulty']}"
}}
MESSAGE assistant {{
    "role": "assistant", 
    "content": "Question: {example['question']}\\n\\nIdeal Answer: {example['ideal_answer']}\\n\\nExpected Keywords: {', '.join(example.get('expected_keywords', []))}"
}}
"""
        
        # Save modelfile
        with open('InterviewExpert.modelfile', 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        self.logger.info("Modelfile created successfully")
        return 'InterviewExpert.modelfile'
    
    def train_ollama_model(self):
        """Train the model using Ollama"""
        self.logger.info("Starting Ollama model training...")
        
        try:
            # Check if Ollama is running
            models = ollama.list()
            self.logger.info("Ollama connection successful")
            
            # Prepare training data
            training_data = self.prepare_training_prompts()
            
            # Create modelfile
            modelfile_path = self.create_modelfile(training_data)
            
            # Create training dataset file
            self.create_training_dataset(training_data)
            
            # Train the model
            self.logger.info("Creating model...")
            response = ollama.create(
                model=self.model_name,
                modelfile=modelfile_path
            )
            
            self.logger.info(f"Model training response: {response}")
            
            # Verify model creation
            models = ollama.list()
            if any(model['name'] == self.model_name for model in models['models']):
                self.logger.info(f"✅ Model '{self.model_name}' trained successfully!")
                return True
            else:
                self.logger.error("❌ Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error training Ollama model: {e}")
            return False
    
    def create_training_dataset(self, training_data):
        """Create a training dataset file for advanced training"""
        self.logger.info("Creating training dataset...")
        
        dataset = []
        for example in training_data:
            # Create conversation format for training
            conversation = [
                {
                    "role": "user",
                    "content": f"Generate a {example['category']} interview question"
                },
                {
                    "role": "assistant",
                    "content": f"Question: {example['question']}\n\nCategory: {example['category']}\n\nIdeal Answer: {example['ideal_answer']}\n\nExpected Keywords: {', '.join(example.get('expected_keywords', []))}"
                }
            ]
            dataset.append({"messages": conversation})
        
        # Save dataset
        with open('interview_training_dataset.jsonl', 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Training dataset created with {len(dataset)} examples")
    
    def evaluate_model(self, test_questions=10):
        """Evaluate the trained model"""
        self.logger.info("Evaluating model...")
        
        try:
            evaluation_results = []
            
            # Sample test questions
            test_samples = self.data.sample(min(test_questions, len(self.data)))
            
            for _, sample in test_samples.iterrows():
                category = sample['category']
                
                # Test question generation
                prompt = f"Generate a {category} interview question"
                response = ollama.generate(model=self.model_name, prompt=prompt)
                
                evaluation_results.append({
                    'category': category,
                    'prompt': prompt,
                    'response': response['response'],
                    'expected_question': sample['question']
                })
                
                time.sleep(1)  # Rate limiting
            
            # Save evaluation results
            with open('model_evaluation.json', 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2)
            
            self.logger.info("Model evaluation completed")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            return None
    
    def train_ml_classifier(self):
        """Train traditional ML classifiers for question categorization"""
        self.logger.info("Training ML classifier for question categorization...")
        
        try:
            # Prepare features and labels
            questions = self.data['question'].tolist()
            categories = self.data['category'].tolist()
            
            # Vectorize questions
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            X = self.vectorizer.fit_transform(questions)
            y = categories
            
            # Train classifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            self.classifier.fit(X, y)
            
            # Save the models
            joblib.dump(self.vectorizer, 'question_vectorizer.pkl')
            joblib.dump(self.classifier, 'category_classifier.pkl')
            
            self.logger.info("ML classifier trained and saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML classifier: {e}")
            return False
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        self.logger.info("Starting complete training pipeline...")
        
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Train ML classifier
            self.train_ml_classifier()
            
            # Step 3: Train Ollama model
            ollama_success = self.train_ollama_model()
            
            # Step 4: Evaluate model
            if ollama_success:
                evaluation_results = self.evaluate_model()
                self.logger.info("Model evaluation completed")
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return ollama_success
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return False

# Usage
if __name__ == "__main__":
    trainer = OllamaInterviewTrainer()
    trainer.run_full_training()