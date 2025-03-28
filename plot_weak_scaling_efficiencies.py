import matplotlib.pyplot as plt
import numpy as np

def plot_line_chart(x_values, y_values_light, y_values_fat):
    """
    Plots a line chart with given x and y values for weak scaling efficiency
    for both Light and Fat clusters.
    
    Parameters:
    x_values (list of int/float): Number of processes (cores).
    y_values_light (list of int/float): Efficiency values (T(1)/T(n)) for Light cluster.
    y_values_fat (list of int/float): Efficiency values (T(1)/T(n)) for Fat cluster.
    """
    # Plot the efficiency line for Light cluster
    plt.plot(x_values, y_values_light, marker='o', linestyle='-', color='b', 
             label='Light Cluster')

    # Plot the efficiency line for Fat cluster
    plt.plot(x_values, y_values_fat, marker='s', linestyle='--', color='r', 
             label='Fat Cluster')

    # Add ideal efficiency line
    plt.axhline(y=1.0, color='gray', linestyle=':', label='Ideal Efficiency')

    # Labels and title with LaTeX-style formatting
    plt.xlabel('Number of Processes ($n$)')
    plt.ylabel('Efficiency ($E(n) = T(1) / T(n)$)')
    plt.title('Weak Scaling Efficiency vs. Number of Processes\n(64 images/core)')

    # Set Y-axis ticks
    plt.yticks(np.arange(0.8, 1.15, 0.05))

    # Grid, legend, and layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig('weak_scaling_efficiency.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# Data from your weak scaling results
x_points = [1, 2, 4, 8, 16]  # Number of processes
y_points_light = [1.0, 0.85, 0.81, 0.83, 0.84]  # Efficiency values for Light cluster
y_points_fat = [1.0, 1.09, 1.11, 1.0, 0.99]    # Efficiency values for Fat cluster

# Generate the plot
plot_line_chart(x_points, y_points_light, y_points_fat)