import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    try:
        # Initialize OpenAI client
        client = OpenAI()
        
        # Test API connection
        models = client.models.list()
        print("Successfully connected to OpenAI!")
        print("\nAvailable models:")
        for model in models:
            print(f"- {model.id}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 