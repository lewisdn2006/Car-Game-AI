"""
Simple AI Training Script - No command line arguments needed
Just run this script to start training the AI!
"""

import sys
import os

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from ai_training import train_ai, test_ai
        
        print("Car Racing AI - Reinforcement Learning")
        print("=====================================")
        print("Welcome! This will train an AI to play the car racing game.")
        print("")
        
        # Ask user what they want to do
        print("What would you like to do?")
        print("1. Train a new AI (recommended for first time)")
        print("2. Continue training existing AI")
        print("3. Test existing AI")
        
        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3")
        
        if choice == '3':
            # Test mode
            print("\nTesting the AI...")
            print("Make sure you have a trained model file (car_racing_ai_model.pth)")
            episodes = int(input("How many test episodes? (default 5): ") or "5")
            test_ai(episodes=episodes, render=True)
            
        else:
            # Training mode
            if choice == '1':
                print("\nStarting fresh training...")
                # Delete existing model if user wants fresh start
                model_path = "car_racing_ai_model.pth"
                if os.path.exists(model_path):
                    delete = input("Delete existing model and start fresh? (y/n): ").strip().lower()
                    if delete == 'y':
                        os.remove(model_path)
                        print("Existing model deleted.")
            else:
                print("\nContinuing training from existing model...")
            
            # Get training parameters
            episodes = int(input("How many episodes to train? (default 500): ") or "500")
            
            render_choice = input("Show game window during training? (y/n, default y): ").strip().lower()
            render = render_choice != 'n'
            
            if not render:
                print("Running without rendering for faster training...")
            
            print(f"\nStarting training for {episodes} episodes...")
            print("Press Ctrl+C to stop training early")
            print("")
            
            # Start training (use save_freq=25 for shorter runs)
            train_ai(episodes=episodes, render=render, save_freq=25)
            
            print("\nTraining completed!")
            print("Model saved as 'car_racing_ai_model.pth'")
            print("You can now test the AI by running this script again and choosing option 3.")
    
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install torch numpy matplotlib pygame")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Progress has been saved!")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
