from ase.io import read, write
import numpy as np
import sys

reference = np.load("nequip-data.npz")
ref_forces = reference["force"]
ref_energy = reference["energy"]

print(ref_energy)
print(reference.files)

frame = read("data.xyz")
frame.wrap()
write("water.data", frame, "lammps-data")

# np.savetxt(    sys.stdout, np.concatenate((frame.get_positions(), ref_forces), axis=1), "%.5f")

result = read("output.dump", format="lammps-dump-text")
print(np.linalg.norm(result.get_forces() - ref_forces))
