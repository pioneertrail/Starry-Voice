"""
Script to test the live flow of the voice chat application.
"""
import os
import logging
import sys
from datetime import datetime
from voice_chat import main

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging to both file and console
log_filename = f"logs/live_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_live_test():
    """Run a live test of the voice chat application."""
    logging.info("Starting live test of voice chat application")
    logging.info("=" * 50)
    
    try:
        print("\nWelcome to the Voice Chat Live Test!")
        print("This test will run for 3 iterations.")
        print("Speak clearly when prompted, and watch the logs for details.")
        print("\nPress Enter to begin...")
        input()
        
        iterations = main(max_iterations=3)
        
        logging.info(f"Test completed successfully with {iterations} iterations")
        logging.info("=" * 50)
        
        print(f"\nTest completed! Check {log_filename} for detailed logs.")
        
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
        print("\nTest interrupted. Check the logs for details.")
    except Exception as e:
        logging.error(f"Test failed: {str(e)}", exc_info=True)
        print(f"\nTest failed: {str(e)}")
        print("Check the logs for details.")

if __name__ == "__main__":
    run_live_test() 