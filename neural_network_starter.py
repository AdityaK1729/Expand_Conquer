"""
Neural Network Starter Code for Expand and Conquer
This provides a foundation for building an AI using deep reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ExpandConquerNN(nn.Module):
    """
    Neural network for Expand and Conquer
    Architecture: CNN-based policy-value network
    """
    def __init__(self, board_size, num_filters=64):
        super(ExpandConquerNN, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm2d(num_filters)
        
        # Calculate flattened size
        flatten_size = num_filters * board_size * board_size
        
        # Policy head (outputs move probabilities)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head (outputs position evaluation)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # x shape: (batch, 3, board_size, board_size)
        # 3 channels: empty, blue, red
        
        # Shared representation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class GameStateEncoder:
    """Encode game state into neural network input format"""
    
    @staticmethod
    def encode_state(board, current_player):
        """
        Encode board state into 3-channel tensor
        Channel 0: Empty squares
        Channel 1: Current player's pieces
        Channel 2: Opponent's pieces
        """
        rows, cols = board.shape
        encoded = np.zeros((3, rows, cols), dtype=np.float32)
        
        # Empty squares
        encoded[0] = (board == 0).astype(np.float32)
        
        # Current player (encode as 1, opponent as 2)
        encoded[1] = (board == current_player).astype(np.float32)
        encoded[2] = (board == (3 - current_player)).astype(np.float32)  # 3-1=2, 3-2=1
        
        return encoded
    
    @staticmethod
    def encode_move(move, board_size):
        """
        Encode move into action index
        For simplicity, map expansion target cells to action indices
        """
        if move['type'] == 'expand':
            # Use first expansion cell as representative
            r, c = min(move['expansion'])
            return r * board_size + c
        elif move['type'] == 'place':
            r, c = move['cell']
            return r * board_size + c
        return 0

class ReplayBuffer:
    """Store and sample training experiences"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class SelfPlayTrainer:
    """Train network through self-play"""
    
    def __init__(self, board_size, lr=0.001):
        self.network = ExpandConquerNN(board_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.board_size = board_size
    
    def select_move(self, game, epsilon=0.1):
        """
        Select move using epsilon-greedy policy
        epsilon: probability of random move (exploration)
        """
        legal_moves = game.get_legal_moves(game.current_player)
        
        if random.random() < epsilon or len(legal_moves) == 0:
            # Random move (exploration)
            return random.choice(legal_moves) if legal_moves else None
        
        # Get network prediction
        state = GameStateEncoder.encode_state(game.board, game.current_player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, _ = self.network(state_tensor)
        
        # Mask illegal moves
        policy_probs = torch.exp(policy_logits).squeeze().numpy()
        legal_mask = np.zeros(self.board_size * self.board_size)
        
        for move in legal_moves:
            action_idx = GameStateEncoder.encode_move(move, self.board_size)
            legal_mask[action_idx] = 1
        
        masked_probs = policy_probs * legal_mask
        if masked_probs.sum() == 0:
            return random.choice(legal_moves)
        
        masked_probs /= masked_probs.sum()
        
        # Sample from probability distribution
        action_idx = np.random.choice(len(masked_probs), p=masked_probs)
        
        # Find corresponding move
        for move in legal_moves:
            if GameStateEncoder.encode_move(move, self.board_size) == action_idx:
                return move
        
        return random.choice(legal_moves)
    
    def play_game(self, game, epsilon=0.1):
        """
        Play one game of self-play
        Returns: game trajectory
        """
        trajectory = []
        
        while not game.is_game_over():
            # Current state
            state = GameStateEncoder.encode_state(game.board, game.current_player)
            current_player = game.current_player
            
            # Select and make move
            move = self.select_move(game, epsilon)
            if move is None:
                break
            
            action_idx = GameStateEncoder.encode_move(move, self.board_size)
            game.make_move(move)
            
            # Next state
            next_state = GameStateEncoder.encode_state(game.board, game.current_player)
            done = game.is_game_over()
            
            # Reward: +1 for win, -1 for loss, 0 otherwise
            if done:
                winner = game.get_winner()
                reward = 1.0 if winner == current_player else -1.0
            else:
                reward = 0.0
            
            trajectory.append((state, action_idx, reward, next_state, done, current_player))
        
        return trajectory
    
    def train_batch(self, batch_size=32):
        """Train on a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Forward pass
        policy_logits, values = self.network(states)
        
        # Policy loss: cross-entropy with selected actions
        policy_loss = F.nll_loss(policy_logits, actions)
        
        # Value loss: MSE with actual rewards
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_games=1000, epsilon_start=0.5, epsilon_end=0.1, epsilon_decay=0.995):
        """
        Main training loop
        """
        from expand_conquer_game import ExpandConquerGame  # Import your game
        
        epsilon = epsilon_start
        losses = []
        
        for game_num in range(num_games):
            # Create new game
            game = ExpandConquerGame(self.board_size, self.board_size, variant="normal")
            
            # Play game and collect trajectory
            trajectory = self.play_game(game, epsilon)
            
            # Assign rewards to all states in trajectory (Monte Carlo)
            winner = game.get_winner() if game.is_game_over() else None
            
            for state, action, _, next_state, done, player in trajectory:
                if winner:
                    reward = 1.0 if winner == player else -1.0
                else:
                    reward = 0.0
                self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train on batch
            if len(self.replay_buffer) >= 32:
                loss = self.train_batch()
                if loss:
                    losses.append(loss)
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Logging
            if (game_num + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:]) if losses else 0
                print(f"Game {game_num + 1}/{num_games}, Epsilon: {epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
        
        return self.network

# Example usage
if __name__ == "__main__":
    print("Expand and Conquer Neural Network Trainer")
    print("=" * 50)
    
    # Initialize trainer
    board_size = 4  # Start with small boards
    trainer = SelfPlayTrainer(board_size=board_size, lr=0.001)
    
    print(f"Training on {board_size}×{board_size} boards")
    print("This will take some time...")
    
    # Train
    trained_network = trainer.train(num_games=1000)
    
    # Save model
    torch.save(trained_network.state_dict(), 'expand_conquer_model.pth')
    print("\nModel saved to 'expand_conquer_model.pth'")
    
    print("\nNext steps:")
    print("1. Evaluate the trained model against random/greedy players")
    print("2. Increase board size gradually")
    print("3. Implement MCTS for stronger play")
    print("4. Add more sophisticated reward shaping")
