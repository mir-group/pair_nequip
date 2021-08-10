# LAMMPS pair style for NequIP

This pair style allows you to use NequIP models in LAMMPS simulations.

*Note: MPI is not supported due to the message-passing nature of the network.*

## Usage in LAMMPS

```
pair_style	nequip
pair_coeff	* * deployed.pth
```
where `deployed.pth` is the filename of your trained model.

## Building LAMMPS with this pair style

### Download LAMMPS
```bash
git clone --depth 1 git@github.com:lammps/lammps
```
or your preferred method.
(`--depth 1` prevents the entire history of the LAMMPS repository from being downloaded.)

### Download this repository
```bash
git clone git@github.com:mir-group/pair_nequip
```

### Patch LAMMPS
#### Automatically
From the `pair_nequip` directory, run:
```bash
./patch_lammps.sh /path/to/lammps/
```

#### Manually
First copy the source files of the pair style:
```bash
cp /path/to/pair_nequip/*.cpp /path/to/lammps/src/
cp /path/to/pair_nequip/*.h /path/to/lammps/src/
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
CMake will look for MKL and, optionally, CUDA and cuDNN. You may have to explicitly provide the path for your CUDA installation (e.g. `-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda/`) and your MKL installation (e.g. `-DMKL_INCLUDE_DIR=/usr/include/`).

Note that the CUDA that comes with PyTorch when installed with `conda` (the `cudatoolkit` package) may not be sufficient (see [here](https://github.com/pytorch/extension-cpp/issues/26), for example) and you may have to install full CUDA seperately.

Pay attention to warnings and error messages.

### Build LAMMPS
```bash
make -j$(nproc)
```
This gives `lammps/build/lmp`, which can be run as usual with `/path/to/lmp -in in.script`. If you specify `-DCMAKE_INSTALL_PREFIX=/somewhere/in/$PATH` (the default is `$HOME/.local`), you can do `make install` and just run `lmp -in in.script`.

