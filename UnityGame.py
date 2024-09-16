import asyncio
import websockets
import json
from RL_BuildUP_v2 import GhostsEnv, QLearningAgent

env = GhostsEnv()
agent = QLearningAgent(env)

# Load a pre-trained model if available
agent.load_model('trained_ghost_model.pkl')

async def game_loop(websocket, path):
    state = env.reset()
    done = False
    player = 0  # Human player starts

    while not done:
        print(f"Player {player}'s turn.")
        
        if player == 0:  # Human player's turn
            legal_actions = env.legal_placement_actions(player) if env.phase == 0 else env.legal_movement_actions(player)
            print(f"Legal actions for player {player}: {legal_actions}")
            
            # Send legal actions to Unity
            await websocket.send(json.dumps({
                "type": "legal_actions",
                "actions": [{"action": action[0], "x": action[1], "y": action[2]} for action in legal_actions]
            }))

            # Receive action from Unity
            response = await websocket.recv()
            action_data = json.loads(response)
            action = (action_data["action"], action_data["x"], action_data["y"], player)
            print(f"Received action from Unity: {action}")
        else:  # AI's turn
            action = agent.select_action(state, player)
            print(f"AI action: {action}")

        # Perform the action
        next_state, reward, done, _ = env.step(action)
        print(f"Action performed. Reward: {reward}, Done: {done}")

        # Send the AI's action to Unity if it's the AI's turn
        if player == 1:
            await websocket.send(json.dumps({
                "type": "ai_action",
                "action": {"action": action[0], "x": action[1], "y": action[2]}
            }))
            print(f"Sent AI action to Unity: {action}")

        state = next_state
        player = 1 - player  # Switch players

    # Game over, send result to Unity
    await websocket.send(json.dumps({
        "type": "game_over",
        "winner": "Human" if done == 0 else "AI" if done == 1 else "Draw"
    }))
    print("Game over. Result sent to Unity.")

start_server = websockets.serve(game_loop, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()