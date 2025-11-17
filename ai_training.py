import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pygame
import sys
import os

# Import the game classes
from Car_driving import Car, Track, Game

class DQN(nn.Module):
    """Deep Q-Network for the racing car AI"""
    
    def __init__(self, input_size=10, hidden_size=128, output_size=3):
        super(DQN, self).__init__()
        # Input: speed (1) + angle (1) + 8 ray distances (8) = 10 features
        # Output: 3 actions (forward, left, right)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class CarRacingAI:
    """Reinforcement Learning AI for the car racing game"""
    
    def __init__(self, state_size=10, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN networks
        self.q_network = DQN(state_size, 128, action_size).to(self.device)
        self.target_network = DQN(state_size, 128, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Keep higher minimum exploration
        self.epsilon_decay = 0.998  # Slower decay to explore more
        self.target_update_freq = 100  # Update target network every 100 episodes
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def get_state(self, car):
        """Extract state features from the car"""
        # Normalize values for better learning
        speed = car.speed / car.max_speed  # Normalize speed to [-1, 1]
        
        # Normalize angle to [-1, 1] range
        angle = (car.angle % 360) / 360.0
        
        # Normalize ray distances (assuming max useful distance is around 300)
        ray_distances = np.array(car.ray_distances) / 300.0
        ray_distances = np.clip(ray_distances, 0, 1)  # Clip to [0, 1]
        
        # Combine all features
        state = np.array([speed, angle] + ray_distances.tolist(), dtype=np.float32)
        return state
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def action_to_keys(self, action):
        """Convert action index to pygame key presses"""
        # Action mapping: 0=forward, 1=left, 2=right
        keys = {
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False
        }
        
        if action == 0:  # Forward
            keys[pygame.K_UP] = True
        elif action == 1:  # Left
            keys[pygame.K_LEFT] = True
        elif action == 2:  # Right
            keys[pygame.K_RIGHT] = True
            
        return keys
    
    def calculate_reward(self, car, track, prev_checkpoint_count, action, collision_occurred):
        """Calculate reward based on game state"""
        reward = 0
        
        # Moderate penalty for collision (not too harsh to encourage exploration)
        if collision_occurred:
            reward -= 50  # Reduced from 100
            return reward
        
        # Large reward for hitting checkpoint (progress is key)
        current_checkpoint_count = (track.current_checkpoint) % len(track.checkpoints)
        if current_checkpoint_count != prev_checkpoint_count:
            reward += 300  # Increased from 200 to emphasize progress
            print(f"Checkpoint reward! Current: {current_checkpoint_count}")
        
        # Reward for maintaining good speed (encourage forward movement)
        speed_reward = car.speed * 3  # Increased multiplier
        reward += speed_reward
        
        # Distance from walls reward (safe driving)
        min_distance = min(car.ray_distances)
        if min_distance > 50:
            reward += 2.0  # Bonus for safe distance
        elif min_distance < 30:
            reward -= (30 - min_distance) * 0.3  # Reduced penalty
        
        # Strong reward for forward movement
        if action == 0 and car.speed > 0:
            reward += 2  # Increased from 1
        
        # Smaller time penalty
        reward -= 0.05  # Reduced from 0.1
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_lengths = checkpoint.get('episode_lengths', [])
            print(f"Model loaded from {filepath}")
            return True
        return False

class AITrainingGame(Game):
    """Modified game class for AI training"""
    
    def __init__(self, render=True):
        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
            
        self.render = render
        if self.render:
            self.screen = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption("Car Racing AI Training")
            self.clock = pygame.time.Clock()
        else:
            # Create a dummy surface for non-rendering mode
            self.screen = pygame.Surface((1200, 800))
            self.clock = pygame.time.Clock()
        
        # Game objects
        self.car = Car(216, 420)
        self.track = Track()
        
        self.reset_game()
        
    def reset_game(self):
        """Reset the game to initial state"""
        self.car.reset_position()
        self.track.reset_checkpoints()
        return self.get_state()
    
    def get_state(self):
        """Get current game state"""
        # Update ray distances
        self.car.update_ray_distances(self.track)
        return {
            'car': self.car,
            'track': self.track,
            'checkpoint_count': self.track.current_checkpoint
        }
    
    def step(self, action_keys):
        """Execute one game step with given action"""
        # Store previous state
        prev_x, prev_y = self.car.x, self.car.y
        prev_checkpoint = self.track.current_checkpoint
        
        # Update car with AI action
        collision_occurred = False
        
        # Handle acceleration/deceleration
        if action_keys[pygame.K_UP]:
            self.car.speed = min(self.car.speed + self.car.acceleration, self.car.max_speed)
        elif action_keys[pygame.K_DOWN]:
            self.car.speed = max(self.car.speed - self.car.acceleration, -self.car.max_speed * 0.5)
        else:
            # Apply friction when no input
            if self.car.speed > 0:
                self.car.speed = max(0, self.car.speed - self.car.deceleration)
            elif self.car.speed < 0:
                self.car.speed = min(0, self.car.speed + self.car.deceleration)
        
        # Handle turning (only when moving)
        if abs(self.car.speed) > 0.1:
            turn_modifier = 1.0 if self.car.speed >= 0 else -1.0
            
            if action_keys[pygame.K_LEFT]:
                self.car.turn_speed = max(self.car.turn_speed - 0.5 * turn_modifier, -self.car.max_turn_speed)
            elif action_keys[pygame.K_RIGHT]:
                self.car.turn_speed = min(self.car.turn_speed + 0.5 * turn_modifier, self.car.max_turn_speed)
            else:
                self.car.turn_speed *= self.car.turn_friction
        else:
            self.car.turn_speed = 0
        
        # Apply turning
        speed_factor = abs(self.car.speed) / self.car.max_speed
        self.car.angle += self.car.turn_speed * speed_factor
        
        # Update position
        rad_angle = np.radians(self.car.angle)
        new_x = self.car.x + np.cos(rad_angle) * self.car.speed
        new_y = self.car.y + np.sin(rad_angle) * self.car.speed
        
        # Check collision
        if not self.track.check_rectangle_collision(new_x, new_y, self.car.width, self.car.height, self.car.angle):
            self.car.x = new_x
            self.car.y = new_y
            
            # Check checkpoint collision
            if self.track.check_checkpoint_collision(self.car.x, self.car.y, self.car.width, self.car.height, self.car.angle):
                self.track.hit_checkpoint()
        else:
            # Collision occurred
            collision_occurred = True
            self.car.reset_position()
            self.track.reset_checkpoints()
        
        # Update ray distances
        self.car.update_ray_distances(self.track)
        
        # Update car image
        self.car.image = pygame.transform.rotate(self.car.original_image, -self.car.angle)
        self.car.rect = self.car.image.get_rect(center=(self.car.x, self.car.y))
        
        # Render if enabled
        if self.render:
            # Handle pygame events to prevent crashes
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return {
                        'car': self.car,
                        'track': self.track,
                        'checkpoint_count': self.track.current_checkpoint,
                        'collision': collision_occurred,
                        'prev_checkpoint': prev_checkpoint,
                        'quit_requested': True
                    }
            
            # Draw the game
            self.track.draw(self.screen)
            self.car.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)  # Limit FPS for training
        
        return {
            'car': self.car,
            'track': self.track,
            'checkpoint_count': self.track.current_checkpoint,
            'collision': collision_occurred,
            'prev_checkpoint': prev_checkpoint,
            'quit_requested': False
        }

def train_ai(episodes=1000, render=True, save_freq=100):
    """Train the AI agent"""
    print("Starting AI training...")
    
    # Initialize game and AI
    game = AITrainingGame(render=render)
    ai = CarRacingAI()
    
    # Try to load existing model
    model_path = "car_racing_ai_model.pth"
    if ai.load_model(model_path):
        print("Continuing training from saved model...")
    
    # Training loop
    for episode in range(episodes):
        state_info = game.reset_game()
        state = ai.get_state(state_info['car'])
        total_reward = 0
        steps = 0
        max_steps = 1000  # Prevent infinite episodes
        
        prev_checkpoint_count = state_info['checkpoint_count']
        
        while steps < max_steps:
            # Get action from AI
            action = ai.get_action(state, training=True)
            action_keys = ai.action_to_keys(action)
            
            # Execute action in game
            next_state_info = game.step(action_keys)
            
            # Check if user closed the window
            if next_state_info.get('quit_requested', False):
                print("Training stopped - window closed")
                break
                
            next_state = ai.get_state(next_state_info['car'])
            
            # Calculate reward
            reward = ai.calculate_reward(
                next_state_info['car'], 
                next_state_info['track'], 
                prev_checkpoint_count,
                action,
                next_state_info['collision']
            )
            
            # Check if episode is done
            done = next_state_info['collision'] or steps >= max_steps - 1
            
            # Store experience
            ai.store_experience(state, action, reward, next_state, done)
            
            # Train the network
            ai.train_step()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            prev_checkpoint_count = next_state_info['checkpoint_count']
            
            if done:
                break
        
        # Update target network periodically
        if episode % ai.target_update_freq == 0:
            ai.update_target_network()
        
        # Record episode results
        ai.episode_rewards.append(total_reward)
        ai.episode_lengths.append(steps)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(ai.episode_rewards[-10:]) if ai.episode_rewards else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {ai.epsilon:.3f}, Steps: {steps}")
        
        # Save model periodically
        if episode % save_freq == 0 and episode > 0:
            ai.save_model(model_path)
    
    # Final save
    ai.save_model(model_path)
    
    print("Training completed!")
    return ai

def plot_training_progress(ai):
    """Plot training progress - DISABLED"""
    # Plotting disabled to avoid showing/saving graphs
    pass
    # if not ai.episode_rewards:
    #     return
    #     
    # plt.figure(figsize=(12, 4))
    # 
    # # Plot rewards
    # plt.subplot(1, 3, 1)
    # plt.plot(ai.episode_rewards)
    # plt.title('Episode Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # 
    # # Plot episode lengths
    # plt.subplot(1, 3, 2)
    # plt.plot(ai.episode_lengths)
    # plt.title('Episode Lengths')
    # plt.xlabel('Episode')
    # plt.ylabel('Steps')
    # 
    # # Plot moving average of rewards
    # plt.subplot(1, 3, 3)
    # window = 50
    # if len(ai.episode_rewards) >= window:
    #     moving_avg = np.convolve(ai.episode_rewards, np.ones(window)/window, mode='valid')
    #     plt.plot(moving_avg)
    #     plt.title(f'Moving Average Reward (window={window})')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Average Reward')
    # 
    # plt.tight_layout()
    # plt.savefig('training_progress.png')
    # plt.show()

def test_ai(episodes=10, render=True):
    """Test the trained AI"""
    print("Testing trained AI...")
    
    game = AITrainingGame(render=render)
    ai = CarRacingAI()
    
    # Load trained model
    model_path = "car_racing_ai_model.pth"
    if not ai.load_model(model_path):
        print("No trained model found!")
        return
    
    ai.epsilon = 0  # No exploration during testing
    
    for episode in range(episodes):
        state_info = game.reset_game()
        state = ai.get_state(state_info['car'])
        total_reward = 0
        steps = 0
        max_steps = 2000
        checkpoints_hit = 0
        
        while steps < max_steps:
            action = ai.get_action(state, training=False)
            action_keys = ai.action_to_keys(action)
            
            next_state_info = game.step(action_keys)
            next_state = ai.get_state(next_state_info['car'])
            
            if next_state_info['checkpoint_count'] != state_info['checkpoint_count']:
                checkpoints_hit += 1
                state_info['checkpoint_count'] = next_state_info['checkpoint_count']
            
            state = next_state
            steps += 1
            
            if next_state_info['collision']:
                break
        
        print(f"Test Episode {episode + 1}: Steps: {steps}, Checkpoints: {checkpoints_hit}")

if __name__ == "__main__":
    # Check if user wants to train or test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_ai(episodes=5, render=True)
    else:
        # Training
        try:
            train_ai(episodes=500, render=True, save_freq=50)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
