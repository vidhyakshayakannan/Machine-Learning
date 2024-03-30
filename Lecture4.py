import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36) # Reproduce random values
X = 2 * np.random.rand(100,1)
Y = 4 + 3*X + np.random.randn(100,1)

# Scatter plot of the data points
plt.scatter(X, Y, color='blue', label='Data Points')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data Points')

# Display the plot
plt.show()

