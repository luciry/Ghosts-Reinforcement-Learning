import gym
import numpy as np
from gym import spaces

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import os

import time
import matplotlib.pyplot as plt


# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
PLACE_BLUE = 4
PLACE_RED = 5
CAPTURE_UP = 6
CAPTURE_RIGHT = 7
CAPTURE_DOWN = 8
CAPTURE_LEFT = 9

ACTIONS = [UP, RIGHT, DOWN, LEFT, PLACE_BLUE, PLACE_RED, CAPTURE_UP, CAPTURE_RIGHT, CAPTURE_DOWN, CAPTURE_LEFT]

# Update ACTION_NAMES
ACTION_NAMES = {
    UP: "Move Up",
    RIGHT: "Move Right",
    DOWN: "Move Down",
    LEFT: "Move Left",
    PLACE_BLUE: "Place Blue Piece",
    PLACE_RED: "Place Red Piece",
    CAPTURE_UP: "Capture Up",
    CAPTURE_RIGHT: "Capture Right",
    CAPTURE_DOWN: "Capture Down",
    CAPTURE_LEFT: "Capture Left"
}

# Define possible spaces
EMPTY = 0
PLAYER_BLUE = 1
PLAYER_RED = 2
OPPONENT_BLUE = 3
OPPONENT_RED = 4
PLAYER_MASKED = 5
OPPONENT_MASKED = 6


# Define mappings from space values to names
SPACE_NAMES = {
    EMPTY: "Empty",
    PLAYER_BLUE: "Player Blue",
    PLAYER_RED: "Player Red",
    OPPONENT_BLUE: "Opponent Blue",
    OPPONENT_RED: "Opponent Red"
}

class GhostsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GhostsEnv, self).__init__()

        # Size of the board is size x size
        self.size = 6

        # Action space is tuple: (action, x, y, player)
        # action: 0-3 (move up, right, down, left), 5 (place blue), 6 (place red), 7-10 (capture up, right, down, left)
        self.action_space = spaces.Tuple((spaces.Discrete(11), spaces.Discrete(self.size), spaces.Discrete(self.size), spaces.Discrete(2)))

        # Reward
        self.reward = 0

        # Initialize the game state
        self.reset(0)

        self.move_counter = 0
        self.max_moves = 100

    def reset(self,player=0):
        # Initialize the board
        self.board = np.zeros((self.size, self.size), dtype=int)

        # Reset phase (0: placement, 1: movement)
        self.phase = 0

        # Reset ghost counts
        self.player_blue_count = 0
        self.opponent_blue_count = 0
        self.player_red_count = 0
        self.opponent_red_count = 0
        self.move_counter = 0
        return self.board

        return self.get_masked_state(player)  # Return masked state for player 0
    
    def get_masked_state(self, player):
        masked_state = np.copy(self.board)
        if player == 0:
            # Mask opponent pieces for player 0
            masked_state[masked_state == OPPONENT_BLUE] = OPPONENT_MASKED
            masked_state[masked_state == OPPONENT_RED] = OPPONENT_MASKED
        else:
            # Mask player pieces for player 1
            masked_state[masked_state == PLAYER_BLUE] = PLAYER_MASKED
            masked_state[masked_state == PLAYER_RED] = PLAYER_MASKED
        return masked_state
    
    def legal_placement_actions(self, player, board=None, flipped=False):
        if board is None:
            board = self.board

        if player == 0:
            rows = [4, 5]
        else:
            rows = [0, 1]
            
        # Recalculate counts
        player_blue_count = np.count_nonzero(board == PLAYER_BLUE)
        player_red_count = np.count_nonzero(board == PLAYER_RED)
        opponent_blue_count = np.count_nonzero(board == OPPONENT_BLUE)
        opponent_red_count = np.count_nonzero(board == OPPONENT_RED)

        actions = []
        valid_cells = [(y, x) for y in rows for x in range(1, self.size - 1) if board[y, x] == EMPTY]

        if player == 0:
            if player_blue_count < 4:
                for x, y in valid_cells:
                    if flipped:
                        actions.append((4, x, y, 1))
                    else:
                        actions.append((4, x, y, player))
            if player_red_count < 4:
                for x, y in valid_cells:
                    if flipped:
                        actions.append((5, x, y, 1))
                    else:
                        actions.append((5, x, y, player))
        else:
            if opponent_blue_count < 4:
                for x, y in valid_cells:
                    actions.append((4, x, y, player))
            if opponent_red_count < 4:
                for x, y in valid_cells:
                    actions.append((5, x, y, player))

        return actions



    def legal_movement_actions(self, player, board=None):
        if board is None:
            board = self.board
        actions = []
        for y in range(self.size):
            for x in range(self.size):
                piece = board[x, y]
                if (player == 0 and piece in [PLAYER_BLUE, PLAYER_RED]) or (player == 1 and piece in [OPPONENT_BLUE, OPPONENT_RED]):
                    # Check all four directions for movement and capture
                    if y > 0:
                        if self.board[x, y-1] == EMPTY:
                            actions.append((LEFT, x, y, player))
                        elif player == 0 and self.board[x, y-1] in [OPPONENT_BLUE, OPPONENT_RED]:
                            actions.append((CAPTURE_LEFT, x, y, player))
                        elif player == 1 and self.board[x, y-1] in [PLAYER_BLUE, PLAYER_RED]:
                            actions.append((CAPTURE_LEFT, x, y, player))
                    
                    if y < self.size - 1:
                        if self.board[x, y+1] == EMPTY:
                            actions.append((RIGHT, x, y, player))
                        elif player == 0 and self.board[x, y+1] in [OPPONENT_BLUE, OPPONENT_RED]:
                            actions.append((CAPTURE_RIGHT, x, y, player))
                        elif player == 1 and self.board[x, y+1] in [PLAYER_BLUE, PLAYER_RED]:
                            actions.append((CAPTURE_RIGHT, x, y, player))
                    
                    if x < self.size - 1:
                        if self.board[x+1, y] == EMPTY:
                            if player == 0:
                                actions.append((DOWN, x, y, player))
                            else:
                                actions.append((DOWN, x, y, player))
                        elif player == 0 and self.board[x+1, y] in [OPPONENT_BLUE, OPPONENT_RED]:
                            actions.append((CAPTURE_DOWN, x, y, player))
                        elif player == 1 and self.board[x+1, y] in [PLAYER_BLUE, PLAYER_RED]:
                            actions.append((CAPTURE_DOWN, x, y, player))
                    
                    if x > 0:
                        if self.board[x-1, y] == EMPTY:
                            if player == 0:
                                actions.append((UP, x, y, player))
                            else:
                                actions.append((UP, x, y, player))
                        elif player == 0 and self.board[x-1, y] in [OPPONENT_BLUE, OPPONENT_RED]:
                            actions.append((CAPTURE_UP, x, y, player))
                        elif player == 1 and self.board[x-1, y] in [PLAYER_BLUE, PLAYER_RED]:
                            actions.append((CAPTURE_UP, x, y, player))
        
        return actions

    def step(self, action):
        self.reward = 0
        if self.phase == 0:
            self._place(action)
        else:
            self._take_action(action)
            self.move_counter += 1  # Increment move counter only in movement phase

        player = action[3]
        done = self._is_done(player)
        reward = self._get_reward()

        # Check if move limit is reached
        if self.move_counter >= self.max_moves:
            done = 3  # Use 3 to indicate a draw due to move limit
            reward = -50  # Negative reward for both players


        return self.board, reward, done, {}


    def _place(self, action):
        action, x, y, player = action
        if player == 0:
            if action == 4:
                self.board[x, y] = PLAYER_BLUE
                self.player_blue_count += 1
            elif action == 5:
                self.board[x, y] = PLAYER_RED
                self.player_red_count += 1
        else:
            if action == 4:
                self.board[x, y] = OPPONENT_BLUE
                self.opponent_blue_count += 1
            elif action == 5:
                self.board[x, y] = OPPONENT_RED
                self.opponent_red_count += 1

    def _take_action(self, action):
        action, x, y, player = action
        if action in [UP, DOWN, LEFT, RIGHT]:
            self._move(x, y, action, player)
        elif action in [CAPTURE_UP, CAPTURE_DOWN, CAPTURE_LEFT, CAPTURE_RIGHT]:
            self._capture(x, y, action, player)

    def _move(self, x, y, direction, player):
        new_x, new_y = x, y
        if direction == UP:
            new_x = x - 1
        elif direction == DOWN:
            new_x = x + 1
        elif direction == LEFT:
            new_y = y - 1
        elif direction == RIGHT:
            new_y = y + 1

        if 0 <= new_x < self.size and 0 <= new_y < self.size and self.board[new_x, new_y] == EMPTY:
            self.board[new_x, new_y] = self.board[x, y]
            self.board[x, y] = EMPTY

    def _capture(self, x, y, direction, player):
        new_x, new_y = x, y
        if direction == CAPTURE_UP:
            new_x = x - 1
        elif direction == CAPTURE_DOWN:
            new_x = x + 1
        elif direction == CAPTURE_LEFT:
            new_y = y - 1
        elif direction == CAPTURE_RIGHT:
            new_y = y + 1

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            captured_piece = self.board[new_x, new_y]
            if (player == 0 and captured_piece in [OPPONENT_BLUE, OPPONENT_RED]) or \
               (player == 1 and captured_piece in [PLAYER_BLUE, PLAYER_RED]):
                self.board[new_x, new_y] = self.board[x, y]
                self.board[x, y] = EMPTY
                if captured_piece == OPPONENT_BLUE:
                    self.opponent_blue_count -= 1
                    self.reward = 20 if player == 0 else -20
                elif captured_piece == OPPONENT_RED:
                    self.opponent_red_count -= 1
                    self.reward = -20 if player == 0 else 20
                elif captured_piece == PLAYER_BLUE:
                    self.player_blue_count -= 1
                    self.reward = -20 if player == 0 else 20
                elif captured_piece == PLAYER_RED:
                    self.player_red_count -= 1
                    self.reward = 20 if player == 0 else -20

    def _get_reward(self):
        
        if self.phase == 0:
            return 0  # No reward during placement phase
        
        if self.move_counter >= self.max_moves:
            return -50  # Negative reward for both players in case of a draw

        
        # Small negative reward for each move to encourage efficiency
        return self.reward - 2
    
    def _is_done(self, player):

        if self.phase == 0:
            if self.player_blue_count == 4 and self.player_red_count == 4 and self.opponent_blue_count == 4 and self.opponent_red_count == 4:
                self.phase = 1
                return 2
            return 2
        else:
            if self.player_blue_count == 0 or self.opponent_red_count == 0:
                # Debug:
                if self.player_blue_count == 0:
                    print("Player Blue count is 0")
                if self.opponent_red_count == 0:
                    print("Opponent Red count is 0")
                if player == 0:
                    self.reward -= 500
                    return 1
                else:
                    self.reward += 500
                    return 1
            elif self.player_red_count == 0 or self.opponent_blue_count == 0:
                # Debug:
                if self.player_red_count == 0:
                    print("Player Red count is 0")
                if self.opponent_blue_count == 0:
                    print("Opponent Blue count is 0")
                if player == 0:
                    self.reward += 500
                    return 0
                else:
                    self.reward -= -500
                    return 0
            
            # Check if there are blue player pieces on the opponent's angles
            if self.board[0, 0] == PLAYER_BLUE or self.board[0, self.size - 1] == PLAYER_BLUE:
                # Debug:
                if self.board[0, 0] == PLAYER_BLUE:
                    print("Player Blue at 0, 0")
                if self.board[0, self.size - 1] == PLAYER_BLUE:
                    print("Player Blue at 0, size - 1")

                if player == 0:
                    self.reward += 500
                    return 0
                else:
                    self.reward -= 500
                    return 0
            if self.board[self.size - 1, 0] == OPPONENT_BLUE or self.board[self.size - 1, self.size - 1] == OPPONENT_BLUE:
                if player == 0:
                    self.reward -= 500
                    return 1
                else:
                    self.reward += 500
                    return 1

            if self.move_counter >= self.max_moves:
                self.reward = -50  # Negative reward for both players
                return 3  # Use 3 to indicate a draw due to move limit
        
            return 2
            
                

    def render(self, mode='human'):
        print("\n " + " ".join([str(i) for i in range(self.size)]))
        for y in range(self.size):
            row = [str(y)]
            for x in range(self.size):
                if self.board[y, x] == EMPTY:
                    row.append(".")
                elif self.board[y, x] == PLAYER_BLUE:
                    row.append("B")
                elif self.board[y, x] == PLAYER_RED:
                    row.append("R")
                elif self.board[y, x] == OPPONENT_BLUE:
                    row.append("b")
                elif self.board[y, x] == OPPONENT_RED:
                    row.append("r")
            print(" ".join(row))
        print()
    
    def close(self):
        pass



physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DQNAgent:
    def __init__(self, env, alpha=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.state_size = env.size * env.size * 7
        self.action_size = 11
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = alpha
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.opponent_model = None  # Add this line
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def _build_model(self):
        model = Sequential([
            Flatten(input_shape=(self.env.size, self.env.size, 7)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))       
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, player, use_epsilon=True):
        if player == 1:
            # Opponent's turn
            # [Existing code for opponent's action]
            pass  # You can keep the existing code here

        # Player's turn
        legal_actions = self.env.legal_placement_actions(player) if self.env.phase == 0 else self.env.legal_movement_actions(player)
        if not legal_actions:
            return None  # No legal actions available

        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)

        state_input = self._preprocess_state(state)
        state_input = tf.convert_to_tensor(state_input, dtype=tf.float32)
        state_input = tf.expand_dims(state_input, 0)
        act_values = self.model.predict(state_input, verbose=0)
        legal_act_values = [act_values[0][action[0]] for action in legal_actions]
        return legal_actions[np.argmax(legal_act_values)]


    
    def opponent_act(self, state, player):
        flipped_state = self.flip_state(state)
        # Generate legal actions for player 0 on the flipped state
        if self.env.phase == 0:
            legal_actions = self.env.legal_placement_actions(0, flipped_state, flipped=True)
        else:
            legal_actions = self.env.legal_movement_actions(0, flipped_state)
        if not legal_actions:
            return None
        state_input = self._preprocess_state(flipped_state)
        state_input = tf.convert_to_tensor(state_input, dtype=tf.float32)
        state_input = tf.expand_dims(state_input, 0)
        act_values = self.opponent_model.predict(state_input, verbose=0)
        legal_act_values = [act_values[0][action[0]] for action in legal_actions]
        best_action = legal_actions[np.argmax(legal_act_values)]
        action = self.flip_action(best_action)
        return action


    def flip_state(self, state):
        flipped_state = np.flipud(np.fliplr(state))
        remapped_state = np.copy(flipped_state)
        
        # Map from original IDs to remapped IDs
        id_mapping = {
            PLAYER_BLUE: OPPONENT_BLUE,
            PLAYER_RED: OPPONENT_RED,
            OPPONENT_BLUE: PLAYER_BLUE,
            OPPONENT_RED: PLAYER_RED,
            PLAYER_MASKED: OPPONENT_MASKED,
            OPPONENT_MASKED: PLAYER_MASKED,
            EMPTY: EMPTY
        }
        
        for original_id, remapped_id in id_mapping.items():
            remapped_state[flipped_state == original_id] = remapped_id

        return remapped_state



    def flip_action(self, action):
        action_type, x, y, player = action
        flipped_action_type = self.flip_action_type(action_type)
        flipped_x = self.env.size - 1 - x
        flipped_y = self.env.size - 1 - y
        return (flipped_action_type, flipped_x, flipped_y, player)

    def flip_action_type(self, action_type):
        flip_mapping = {
            # Movement actions
            0: 2,  # UP becomes DOWN
            1: 3,  # RIGHT becomes LEFT
            2: 0,  # DOWN becomes UP
            3: 1,  # LEFT becomes RIGHT
            # Placement actions remain the same
            4: 4,  # PLACE BLUE
            5: 5,  # PLACE RED
            # Capture actions
            6: 8,  # CAPTURE UP becomes CAPTURE DOWN
            7: 9,  # CAPTURE RIGHT becomes CAPTURE LEFT
            8: 6,  # CAPTURE DOWN becomes CAPTURE UP
            9: 7,  # CAPTURE LEFT becomes CAPTURE RIGHT
            10: 10  # If there's an action 10, adjust accordingly
        }
        return flip_mapping.get(action_type, action_type)


    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            target = self.model(states, training=True)
            next_q_values = self.target_model(next_states, training=False)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            masks = tf.one_hot(actions, self.action_size)
            q_values = tf.reduce_sum(target * masks, axis=1)
            loss = tf.keras.losses.MSE(target_q_values, q_values)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


    def _preprocess_state(self, state):
        encoded_state = np.zeros((self.env.size, self.env.size, 7))
        for i in range(self.env.size):
            for j in range(self.env.size):
                piece_id = int(state[i, j])
                encoded_state[i, j, piece_id] = 1
        return encoded_state


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array([self._preprocess_state(state) for state in states])
        next_states = np.array([self._preprocess_state(state) for state in next_states])
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor([action[0] for action in actions], dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        loss = self.train_step(states, actions, rewards, next_states, dones)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        #print(f"Replay completed. New epsilon: {self.epsilon:.4f}, Loss: {loss:.4f}")

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        if not name.endswith('.weights.h5'):
            name += '.weights.h5'
        self.model.save_weights(name)



    def train(self, n_segments, k_games_per_segment, batch_size=16):
        start_time = time.time()
        win_rates = []
        draw_rates = []
        loss_rates = []
        total_wins = 0
        total_draws = 0
        total_losses = 0
        outcomes = []
        for segment in range(n_segments):
            if segment == 0:
                self.opponent_model = None  # Use random opponent initially
            else:
                # Clone the agent's model to be the opponent
                self.opponent_model = self.clone_model()
                self.epsilon = 0.85
            for game in range(k_games_per_segment):
                state = self.env.reset()
                done = 2
                player = 0
                total_reward = 0
                steps = 0
                print_game = (game == 0)  # Print the first game of each segment
                while True:
                    if player == 0:
                        action = self.act(state, player)
                    else:
                        if self.opponent_model is None:
                            # Random opponent
                            legal_actions = self.env.legal_placement_actions(player) if self.env.phase == 0 else self.env.legal_movement_actions(player)
                            action = random.choice(legal_actions) if legal_actions else None
                        else:
                            action = self.opponent_act(state, player)
                    if action is None:
                        print(f"No legal actions for player {player}. Ending game.")
                        if print_game:
                            print("Final State:")
                            self.env.render()
                        # Get the actual game status
                        done = self.env._is_done(player)
                        # If the game isn't over, the player with no moves loses
                        if done > 1:
                            done = 1 if player == 0 else 0
                        break

                    next_state, reward, done, _ = self.env.step(action)

                    if print_game:
                        print(f"Player {player} took action {action}")
                        self.env.render()

                    if player == 0:
                        self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1

                    if done <= 1 or done == 3:
                        break

                    player = 1 - player

                    if len(self.memory) > batch_size and player == 0:
                        self.replay(batch_size)
                elapsed_time = time.time() - start_time
                print(f"Segment: {segment + 1}, Game: {game + 1}, Steps: {steps}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}, Elapsed Time: {elapsed_time:.2f}s")

                # Record the outcome
                if done == 0:  # Agent wins
                    total_wins += 1
                    outcomes.append(1)
                    print("Agent wins!")
                elif done == 1:  # Agent loses
                    total_losses += 1
                    outcomes.append(0)
                    print("Agent loses.")
                elif done == 3:  # Draw
                    total_draws += 1
                    outcomes.append(0.5)
                    print("Game ended in a draw.")
                else:
                    outcomes.append(0)
                    print("Unexpected game outcome.")

                # Calculate cumulative rates
                total_games_played = len(outcomes)
                win_rate = total_wins / total_games_played
                draw_rate = total_draws / total_games_played
                loss_rate = total_losses / total_games_played
                win_rates.append(win_rate)
                draw_rates.append(draw_rate)
                loss_rates.append(loss_rate)

            self.update_target_model()
        self.save("dqn_ghost_model.h5")
        print("Training completed.")
        return win_rates, draw_rates, loss_rates



    
    def clone_model(self):
        cloned_model = self._build_model()
        cloned_model.set_weights(self.model.get_weights())
        return cloned_model



    def plot_win_rates(self, win_rates, eval_interval):
        plt.figure(figsize=(10, 6))
        plt.plot(range(eval_interval, len(win_rates) * eval_interval + 1, eval_interval), win_rates)
        plt.title('Win Rate Against Random Agent Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.show()

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def plot_rates(self, win_rates, draw_rates, loss_rates, window_size=10):
        episodes = list(range(1, len(win_rates) + 1))
        win_rates_smooth = self.moving_average(win_rates, window_size)
        draw_rates_smooth = self.moving_average(draw_rates, window_size)
        loss_rates_smooth = self.moving_average(loss_rates, window_size)
        episodes_smooth = episodes[window_size - 1:]

        plt.figure(figsize=(10, 6))
        plt.plot(episodes_smooth, win_rates_smooth, label='Win Rate')
        plt.plot(episodes_smooth, draw_rates_smooth, label='Draw Rate')
        plt.plot(episodes_smooth, loss_rates_smooth, label='Loss Rate')
        plt.title('Smoothed Win/Draw/Loss Rates Over Episodes')
        plt.xlabel('Game Number')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    def evaluate_against_random(self, num_games):
        outcomes = []
        for game in range(num_games):
            state = self.env.reset()
            done = 2
            player = 0
            while True:
                if player == 0:
                    # Agent's turn
                    action = self.act(state, player, use_epsilon=False)
                else:
                    # Random opponent
                    legal_actions = self.env.legal_placement_actions(player) if self.env.phase == 0 else self.env.legal_movement_actions(player)
                    action = random.choice(legal_actions) if legal_actions else None
                if action is None:
                    print(f"No legal actions for player {player}. Ending game.")
                    # Get the actual game status
                    done = self.env._is_done(player)
                    # If the game isn't over, the player with no moves loses
                    if done > 1:
                        done = 1 if player == 0 else 0
                    break

                next_state, reward, done, _ = self.env.step(action)
                state = next_state

                if done <= 1 or done == 3:
                    break

                player = 1 - player

            # Record the outcome
            if done == 0:  # Agent wins
                outcomes.append(1)
                print(f"Game {game + 1}: Agent wins!")
            elif done == 1:  # Agent loses
                outcomes.append(0)
                print(f"Game {game + 1}: Agent loses.")
            elif done == 3:  # Draw
                outcomes.append(0.5)
                print(f"Game {game + 1}: Game ended in a draw.")
            else:
                outcomes.append(0)
                print(f"Game {game + 1}: Unexpected game outcome.")

        # Calculate cumulative rates
        total_games_played = len(outcomes)
        win_rates = []
        draw_rates = []
        loss_rates = []
        total_wins = total_draws = total_losses = 0
        for i, outcome in enumerate(outcomes, 1):
            if outcome == 1:
                total_wins += 1
            elif outcome == 0.5:
                total_draws += 1
            else:
                total_losses += 1
            win_rates.append(total_wins / i)
            draw_rates.append(total_draws / i)
            loss_rates.append(total_losses / i)

        return win_rates, draw_rates, loss_rates
