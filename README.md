# LAMMPS pair style for NequIP

This pair style allows you to use NequIP models from the [`nequip`](https://github.com/mir-group/nequip) framework in LAMMPS simulations. For more details on NequIP and the Python code, please visit the [`nequip`](https://github.com/mir-group/nequip) repository.

*Please Note: MPI is not supported due to the message-passing nature of the network. For MPI support with large numbers of atoms, please consider our [Allegro model](https://github.com/mir-group/allegro) and corresponding [`pair_allegro`](https://github.com/mir-group/pair_allegro) LAMMPS plugin.*

`pair_nequip` authors: **Anders Johansson**, Albert Musaelian, Lixin Sun.

## Pre-requisites

* PyTorch or LibTorch >= 1.11.0;  please note that at present we have only thoroughly tested 1.11 on NVIDIA GPUs (see [#311 for NequIP](https://github.com/mir-group/nequip/discussions/311#discussioncomment-5129513)) and 1.13 on AMD GPUs, but newer 2.x versions *may* also work. With newer versions, setting the environment variable `PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1` sometimes helps.

## Usage in LAMMPS

```
pair_style	nequip
pair_coeff	* * deployed.pth <type name 1> <type name 2> ...
```
where `deployed.pth` is the filename of your trained, **deployed** model.

The names after the model path `deployed.pth` indicate, in order, the names of the NequIP model's atom types to use for LAMMPS atom types 1, 2, and so on. The number of names given must be equal to the number of atom types in the LAMMPS configuration (not the NequIP model!).
The given names must be consistent with the names specified in the NequIP training YAML in `chemical_symbol_to_type` or `type_names`.

## Building LAMMPS with this pair style

### Download LAMMPS
```bash
git clone --depth=1 https://github.com/lammps/lammps
```
or your preferred method.
(`--depth=1` prevents the entire history of the LAMMPS repository from being downloaded.)

### Download this repository
```bash
git clone https://github.com/mir-group/pair_nequip
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

Pay attention to warnings and error messages.

**MKL:** If `MKL_INCLUDE_DIR` is not found and you are using a Python environment, a simple solution is to run `conda install mkl-include` or `pip install mkl-include` and append:
```
-DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
```
to the `cmake` command if using a `conda` environment, or
```
-DMKL_INCLUDE_DIR=`python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"`
```
if using plain Python and `pip`.

**CUDA:** Note that the CUDA that comes with PyTorch when installed with `conda` (the `cudatoolkit` package) is usually insufficient (see [here](https://github.com/pytorch/extension-cpp/issues/26), for example) and you may have to install full CUDA seperately. A minor version mismatch between the available full CUDA version and the version of `cudatoolkit` is usually *not* a problem, as long as the system CUDA is equal or newer. (For example, PyTorch's requested `cudatoolkit==11.3` with a system CUDA of 11.4 works, but a system CUDA 11.1 will likely fail.)

### Build LAMMPS
```bash
make -j$(nproc)
```
This gives `lammps/build/lmp`, which can be run as usual with `/path/to/lmp -in in.script`. If you specify `-DCMAKE_INSTALL_PREFIX=/somewhere/in/$PATH` (the default is `$HOME/.local`), you can do `make install` and just run `lmp -in in.script`.

## FAQ

1. Q: My simulation is immediately or bizzarely unstable

   A: Please ensure that your mapping from LAMMPS atom types to NequIP atom types, specified in the `pair_coeff` line, is correct.
2. Q: I get the following error:
   ```
    instance of 'c10::Error'
        what():  PytorchStreamReader failed locating file constants.pkl: file not found
   ```

   A: Make sure you remembered to deploy (compile) your model using `nequip-deploy`, and that the path to the model given with `pair_coeff` points to a deployed model `.pth` file, **not** a file containing only weights like `best_model.pth`.
3. Q: I get the following error:
    ```
    Exception: Argument passed to at() was not in the map
    ```

    A: We now require models to have been trained with stress support, which is achieved by replacing `ForceOutput` with `StressForceOutput` in the training configuration. Note that you do not need to train on stress (though it may improve your potential, assuming your stress data is correct and converged). If you desperately wish to keep using a model without stress output, there are two options: 1) Remove lines that look like [these](https://github.com/mir-group/pair_allegro/blob/99036043e74376ac52993b5323f193dee3f4f401/pair_allegro_kokkos.cpp#L332-L343) in your version of `pair_allegro[_kokkos].cpp` 2) Redeploy the model with an updated config file, as described [here](https://github.com/mir-group/nequip/issues/69#issuecomment-1129273665).
