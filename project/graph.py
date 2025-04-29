import re
import matplotlib.pyplot as plt

# Sample log data (replace with your actual log file reading)
log_data = """
SPS: 30
Step 5100: Q1 Loss = 2.321, Q2 Loss = 2.629, Actor Loss = -7.454
Kart crashed! Respawning...
Kart crashed! Respawning...
SPS: 16
Step 5200: Q1 Loss = 2.538, Q2 Loss = 2.489, Actor Loss = -7.399
Kart crashed! Respawning...
SPS: 11
Step 5300: Q1 Loss = 1.927, Q2 Loss = 1.296, Actor Loss = -8.852
SPS: 9
Step 5400: Q1 Loss = 1.059, Q2 Loss = 0.454, Actor Loss = -9.392
Kart crashed! Respawning...
SPS: 7
Step 5500: Q1 Loss = 0.851, Q2 Loss = 1.034, Actor Loss = -10.943
Kart crashed! Respawning...
Kart crashed! Respawning...
SPS: 6
Step 5600: Q1 Loss = 0.944, Q2 Loss = 0.435, Actor Loss = -12.019
Kart crashed! Respawning...
SPS: 5
Step 5700: Q1 Loss = 0.918, Q2 Loss = 0.464, Actor Loss = -13.817
SPS: 5
Step 5800: Q1 Loss = 0.868, Q2 Loss = 0.493, Actor Loss = -15.815
Kart crashed! Respawning...
SPS: 4
Step 5900: Q1 Loss = 0.880, Q2 Loss = 0.898, Actor Loss = -15.778
Kart crashed! Respawning...
SPS: 4
Step 6000: Q1 Loss = 1.472, Q2 Loss = 1.772, Actor Loss = -18.939
Resetting the environment...
Kart crashed! Respawning...
SPS: 3
Step 6100: Q1 Loss = 6.629, Q2 Loss = 4.860, Actor Loss = -19.036
Kart crashed! Respawning...
SPS: 3
Step 6200: Q1 Loss = 6.662, Q2 Loss = 6.502, Actor Loss = -19.550
Kart crashed! Respawning...
SPS: 3
Step 6300: Q1 Loss = 2.065, Q2 Loss = 0.815, Actor Loss = -20.611
Kart crashed! Respawning...
SPS: 3
Step 6400: Q1 Loss = 2.933, Q2 Loss = 2.624, Actor Loss = -21.724
Kart crashed! Respawning...
SPS: 3
Step 6500: Q1 Loss = 1.443, Q2 Loss = 3.369, Actor Loss = -21.996
Kart crashed! Respawning...
Kart crashed! Respawning...
SPS: 2
Step 6600: Q1 Loss = 1.591, Q2 Loss = 1.532, Actor Loss = -24.745
SPS: 2
Step 6700: Q1 Loss = 2.009, Q2 Loss = 1.081, Actor Loss = -24.711
Kart crashed! Respawning...
SPS: 2
Step 6800: Q1 Loss = 6.375, Q2 Loss = 2.497, Actor Loss = -24.673
Kart crashed! Respawning...
SPS: 2
Step 6900: Q1 Loss = 2.482, Q2 Loss = 2.369, Actor Loss = -24.492
Kart crashed! Respawning...
SPS: 2
Step 7000: Q1 Loss = 1.566, Q2 Loss = 1.680, Actor Loss = -26.966
Resetting the environment...
"""

# Parse the log data
steps = []
q1_losses = []
q2_losses = []
actor_losses = []

for line in log_data.split('\n'):
    if line.startswith('Step'):
        # Extract step number and losses using regular expressions
        match = re.search(r'Step (\d+): Q1 Loss = ([\d.-]+), Q2 Loss = ([\d.-]+), Actor Loss = ([\d.-]+)', line)
        if match:
            steps.append(int(match.group(1)))
            q1_losses.append(float(match.group(2)))
            q2_losses.append(float(match.group(3)))
            actor_losses.append(float(match.group(4)))

# Create the plot
plt.figure(figsize=(12, 6))

# Plot Q1 Loss
plt.plot(steps, q1_losses, label='Q1 Loss', color='blue', alpha=0.7)

# Plot Q2 Loss
plt.plot(steps, q2_losses, label='Q2 Loss', color='green', alpha=0.7)

# Plot Actor Loss (absolute value since it's negative)
plt.plot(steps, [abs(x) for x in actor_losses], label='|Actor Loss|', color='red', alpha=0.7)

# Add labels and title
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Training Losses Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust x-axis to show step numbers properly
plt.xticks(steps[::len(steps)//10], rotation=45)

plt.tight_layout()
plt.show()