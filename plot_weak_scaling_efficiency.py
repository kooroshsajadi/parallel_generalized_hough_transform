import matplotlib.pyplot as plt

def plot_line_chart(x_values, y_values):
    """
    Plots a line chart with given x and y values for weak scaling efficiency.
    
    Parameters:
    x_values (list of int/float): Number of processes (cores).
    y_values (list of int/float): Efficiency values (T(1)/T(n)).
    """
    # Plot the efficiency line
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', 
             label='Weak Scaling Efficiency\n(64 images/core)')

    # Labels and title with LaTeX-style formatting
    plt.xlabel('Number of Processes ($n$)')
    plt.ylabel('Efficiency ($E(n) = T(1) / T(n)$)')
    plt.title('Weak Scaling Efficiency vs. Number of Processes')

    # Grid, legend, and layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Data from your weak scaling results
x_points = [1, 2, 4, 8, 16]  # Number of processes
y_points = [1.0, 0.85, 0.81, 0.83, 0.84]  # Efficiency values

# Generate the plot
plot_line_chart(x_points, y_points)