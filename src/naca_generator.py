import numpy as np
import random

class NACAGenerator:
    def __init__(self, num_points=200):
        # Cosine spacing clusters points at the leading/trailing edges for better CFD meshing
        beta = np.linspace(0, np.pi, num_points)
        self.x = (1 - np.cos(beta)) / 2

    def generate_airfoil(self, m, p, t):
        """
        Generates NACA 4-digit airfoil coordinates.
        m: Max camber (0.0 to 0.09)
        p: Position of max camber (0.1 to 0.9)
        t: Max thickness (0.05 to 0.30)
        """
        x = self.x

        # Thickness distribution
        yt = 5 * t * (0.2969 * np.sqrt(x) -
                      0.1260 * x -
                      0.3516 * (x**2) +
                      0.2843 * (x**3) -
                      0.1015 * (x**4))
        
        # Camber line and gradient
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        if m > 0:
            # Front part of the airfoil (x < p)
            front = x < p
            yc[front] = (m / p**2) * (2 * p * x[front] - x[front]**2)
            dyc_dx[front] = (2 * m / p**2) * (p - x[front])

            # Back part of the airfoil
            back = x >= p
            yc[back] = (m / (1-p)**2) * ((1- 2 * p) + 2 * p * x[back] - x[back]**2)
            dyc_dx[back] = (2 * m / (1-p)**2) * (p - x[back])
        
        theta = np.arctan(dyc_dx)

        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine coordinates (trailing edge -> leading edge -> trailing edge)
        X = np.concatenate((xu[::-1], xl[1:]))
        Y = np.concatenate((yu[::-1], yl[1:]))
        
        return X, Y
    
    def export_for_meshing(self, filename, X, Y):
        """Saves the coordinates to a format easily read by Gmsh or blockMesh scripts."""
        coords = np.column_stack((X, Y, np.zeros_like(X))) # Add Z=0 for 2D
        np.savetxt(filename, coords, fmt='%.6f', delimiter=',')
        print(f"Airfoil geometry saved to {filename}")

# --- Test the Factory Logic ---
if __name__ == "__main__":
    # This block now only runs generation/export without needing matplotlib
    generator = NACAGenerator(num_points=150)

    for i in range(3):
        # Randomize within sensible aerodynamic limits
        m = random.uniform(0.0, 0.06)      # 0 to 6% camber
        p = random.uniform(0.2, 0.6)       # Max camber at 20% to 60% chord
        t = random.uniform(0.08, 0.25)     # 8% to 25% thickness

        name = f"NACA_{int(m*100)}{int(p*10)}{int(t*100):02d}"

        X, Y = generator.generate_airfoil(m, p, t)
        generator.export_for_meshing(f"{name}.csv", X, Y)
        
    print("Test generation complete.")