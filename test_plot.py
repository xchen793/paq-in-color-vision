import matplotlib.pyplot as plt
import colour

def plot_cie_1931_chromaticity_diagram():
    # Plot the CIE 1931 Chromaticity Diagram
    plt.figure(figsize=(8, 8))
    colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
    
    # Customize the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('CIE 1931 Chromaticity Diagram')
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Call the function to plot the diagram
plot_cie_1931_chromaticity_diagram()
