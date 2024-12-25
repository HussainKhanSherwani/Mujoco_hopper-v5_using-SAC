import gymnasium as gym
from stable_baselines3 import SAC

# Create the Hopper environment
env = gym.make("Hopper-v5", render_mode="human")  # 'human' allows the environment to be rendered

# Load the trained model
model = SAC.load("sac_hopper_lr_0.01.zip", env=env)  # Replace with your model path if different

# Run the trained agent in the environment
obs, info = env.reset()
for _ in range(250000):  # Run for a set number of timesteps
    action, _states = model.predict(obs, deterministic=True)  # Predict action from the model
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

# Close the environment after the simulation
env.close()
