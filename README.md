# LAMMPS pair style for NEQUIP

This pair style allows you to use NEQUIP models in LAMMPS simulations.

## Usage in LAMMPS

```
pair_style	nequip
pair_coeff	* * deployed.pth
```
where `deployed.pth` is the filename of your trained model.

## Building LAMMPS with this pair style

### Download LAMMPS
```bash
git clone git@github.com:lammps/lammps
```
or your preferred method.

### Download this repository
```bash
git clone git@github.com:mir-group/pair_nequip
```

### Patch LAMMPS
First copy the source files of the pair style:
```bash
cp /path/to/pair_nequip/pair_nequip.* /path/to/lammps/src/
```
Then make the following modifications to `lammps/cmake/CMakeLists.txt`:
- Change `set(CMAKE_CXX_STANDARD 11)` to `set(CMAKE_CXX_STANDARD 14)`
- Append the following lines:
```cmake
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
```

### Configure LAMMPS
If you have PyTorch installed:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```
If you don't have PyTorch installed, you need to download LibTorch from the [PyTorch download page](https://pytorch.org/get-started/locally/). Unzip the downloaded file, then configure LAMMPS:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
```
CMake will look for MKL and, optionally, CUDA and cuDNN. Pay attention to warnings and error messages.

### Build LAMMPS
```bash
make -j$(nproc)
```
This gives `lammps/build/lmp`, which can be run as usual with `/path/to/lmp -in in.script`. If you specify `-DCMAKE_INSTALL_PREFIX=/somewhere/in/$PATH` (the default is `$HOME/.local`), you can do `make install` and just run `lmp -in in.script`.

