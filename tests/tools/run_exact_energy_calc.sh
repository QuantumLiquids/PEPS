#!/bin/bash
# Compile and run the exact energy calculator

echo "Compiling exact energy calculator..."
g++ -O3 -std=c++17 tools/calculate_classical_potts_exact_energies.cpp -o tools/calculate_classical_potts_exact_energies

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running calculator..."
    echo "This may take several minutes for larger systems..."
    cd tools
    ./calculate_classical_potts_exact_energies
    echo "Results saved in tools/potts_exact_energies.txt"
else
    echo "Compilation failed!"
    exit 1
fi