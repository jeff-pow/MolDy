# MolDy (Molecular Dynamics :))

This program offers an efficient solution for simulating the behavior of Argon atoms in a standard canonical ensemble (NVT). By implementing multithreading and the cell lists algorithm, computational efficiency is greatly improved, reducing the computational complexity from the normal pairwise O(N^2) algorithm to an optimized O(N).

### Features:
- Accurate modeling of Argon atoms in the NVT ensemble.
- Visualization: The program generates a visualization of the simulated system, which is saved as a .xyz file. You can use popular visualization software such as VMD (https://www.ks.uiuc.edu/Research/vmd/.) to visualize and analyze the generated data.
- Multithreading: The program takes advantage of multithreading, allowing for faster simulation times on multi-core processors.
- Efficient algorithm: The implementation of the cell lists algorithm significantly improves the simulation's performance by reducing the computational complexity.
