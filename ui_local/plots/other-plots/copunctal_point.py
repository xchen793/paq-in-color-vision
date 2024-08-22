import matplotlib.pyplot as plt
import numpy as np
import colour

# Copunctal point for Protanopia
protan_copunctal_point = (0.747, 1.647)

# Define the chromaticity diagram and boundary points
fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False, axes=ax)
ax.set_title("Protanopia confusion lines and its copunctal point")

# Define the line segment points
copunctal_point = (0.747, 0.253)

starting_points = np.array([
    [0.075, 0.83], [0.02, 0.75], [0.007, 0.65], [0.01, 0.55], 
    [0.02, 0.45], [0.035, 0.35], [0.06, 0.25], [0.09, 0.15], 
    [0.1, 0.25]
])

for start_point in starting_points:
    line_x = [start_point[0], copunctal_point[0]]
    line_y = [start_point[1], copunctal_point[1]]
    ax.plot(line_x, line_y, color='red', linewidth=2)

ax.plot([], [], color='red', linewidth=2, label='Protan confusion Lines')

# Plot the start and copunctal points
ax.plot(copunctal_point[0], copunctal_point[1], 'o', label='Protan Point', color='blue')

ax.legend()
plt.grid(True)
plt.savefig('protanopia_conf_lines(copoint).png', dpi=300)
