import asyncio
import websockets
import json
import numpy as np

from game_classes import GhostsEnv
from game_classes import DQNAgent

# Initialize the environment and agent
env = GhostsEnv()
agent = DQNAgent(env)

# Load the trained model weights
try:
    agent.model.load_weights('dqn_ghost_model.h5.weights.h5')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

async def handler(websocket, path):
    print("Client connected")
    state = env.reset()
    done = False
    player = 0  # 0 for human player, 1 for AI

    while not done:
        if player == 0:
            # Human player's turn, send legal moves to Unity
            legal_actions = env.legal_placement_actions(player) if env.phase == 0 else env.legal_movement_actions(player)
            legal_moves = [{"action": action} for action in legal_actions]
            print(f"Sending legal moves to client: {legal_moves}")
            await websocket.send(json.dumps({"legal_moves": legal_moves}))

            # Receive player's move from Unity
            move = await websocket.recv()
            move = json.loads(move)
            action = tuple(move["action"])
            print(f"Received player's move: {action}")
        else:
            # AI's turn
            action = agent.act(state, player, use_epsilon=False)
            if action is None:
                print("No legal actions for AI. Ending game.")
                break  # No valid moves for AI, game ends
            
            print(f"AI selected action: {action}")
            # Send AI's move to Unity
            await websocket.send(json.dumps({"ai_move": {"action": action}}))
            print(f"Sent AI's move to client: {action}")

        # If no action is found, the game is over
        if action is None:
            print(f"No legal actions for player {player}. Ending game.")
            done = True
            break

        # Apply the action and get the next state, reward, and game status
        next_state, reward, done_flag, _ = env.step(action)
        state = next_state

        if done_flag == 0:  # If done_flag == 0, agent won
            print("Agent won!")
            done = True
        elif done_flag == 1:  # If done_flag == 1, agent lost
            print("Agent lost!")
            done = True
        elif done_flag == 3:  # If done_flag == 3, it's a draw
            print("Game ended in a draw.")
            done = True
        else:
            # Continue the game if not finished
            done = False

        if done:  # Check if the game is over
            print(f"Game over detected. Ending game.")
            break

        # Alternate player (switch turns)
        player = 1 - player

    # Game over, send final state to client
    await websocket.send(json.dumps({"game_over": True}))
    print("Game over sent to client.")

async def main():
    async with websockets.serve(handler, 'localhost', 8765):
        print("Server started at ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
