#!/bin/bash
# patch_lammps.sh /path/to/lammps/

lammps_dir=$1

# Check and produce nice message
if [ ! -f pair_nequip.cpp ]; then
    echo "Please run `patch_lammps.sh` from the `pair_nequip` root directory."
    exit 1
fi

# Check for double-patch
if grep -q "find_package(Torch REQUIRED)" $lammps_dir/cmake/CMakeLists.txt ; then
    echo "This LAMMPS installation _seems_ to already have been patched; please check it!"
    exit 1
fi

echo "Copying files..."
cp *.cpp $lammps_dir/src/
cp *.h $lammps_dir/src/

echo "Updating CMakeLists.txt..."

# Update CMakeLists.txt
sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 14)/" $lammps_dir/cmake/CMakeLists.txt

# Add libtorch
cat >> $lammps_dir/cmake/CMakeLists.txt << "EOF2"

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF2

echo "Done!"