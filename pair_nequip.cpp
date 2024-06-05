/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <pair_nequip.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// Freezing is broken from C++ in <=1.10; so we've dropped support.
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#error "PyTorch version < 1.11 is not supported"
#endif

using namespace LAMMPS_NS;

PairNEQUIP::PairNEQUIP(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "NEQUIP is using device " << device << "\n";

  if(const char* env_p = std::getenv("NEQUIP_DEBUG")){
    std::cout << "PairNEQUIP is in DEBUG mode, since NEQUIP_DEBUG is in env\n";
    debug_mode = 1;
  }
}

PairNEQUIP::~PairNEQUIP(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
  }
}

void PairNEQUIP::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NEQUIP requires atom IDs");

  neighbor->add_request(this, NeighConst::REQ_FULL);

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR,"Pair style NEQUIP requires newton pair off");
}

double PairNEQUIP::init_one(int i, int j)
{
  return cutoff;
}

void PairNEQUIP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(type_mapper, n+1, "pair:type_mapper");

}

void PairNEQUIP::settings(int narg, char ** /*arg*/) {
  // "nequip" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

void PairNEQUIP::coeff(int narg, char **arg) {

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Parse the definition of each atom type
  char **elements = new char*[ntypes+1];
  for (int i = 1; i <= ntypes; i++){
      elements[i] = new char [strlen(arg[i+2])+1];
      strcpy(elements[i], arg[i+2]);
      if (screen) fprintf(screen, "NequIP Coeff: type %d is element %s\n", i, elements[i]);
  }

  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  std::cout << "Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {
    {"config", ""},
    {"nequip_version", ""},
    {"r_max", ""},
    {"n_species", ""},
    {"type_names", ""},
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""},
    {"allow_tf32", ""}
  };
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty()) {
    error->all(FLERR, "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?");
  }

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training")) {
    std::cout << "Freezing TorchScript model...\n";
    model = torch::jit::freeze(model);
  }

  // In PyTorch >=1.11, this is now set_fusion_strategy
  torch::jit::FusionStrategy strategy;
  if (metadata["_jit_fusion_strategy"].empty()) {
    // This is the default used in the Python code
    strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  } else {
    std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
    std::string fusion_type, fusion_depth;
    while(std::getline(strat_stream, fusion_type, ',')) {
      std::getline(strat_stream, fusion_depth, ';');
      strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
    }
  }
  torch::jit::setFusionStrategy(strategy);

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  // std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  // match the type names in the pair_coeff to the metadata
  // to construct a type mapper from LAMMPS type to NequIP atom_types
  int n_species = std::stod(metadata["n_species"]);
  std::stringstream ss;
  ss << metadata["type_names"];
  for (int i = 0; i < n_species; i++){
      char ele[100];
      ss >> ele;
      for (int itype = 1; itype <= ntypes; itype++)
          if (strcmp(elements[itype], ele) == 0)
              type_mapper[itype] = i;
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
        if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0))
            setflag[i][j] = 1;

  if (elements){
      for (int i=1; i<ntypes; i++)
          if (elements[i]) delete [] elements[i];
      delete [] elements;
  }

}

// Force and energy computation
void PairNEQUIP::compute(int eflag, int vflag){
  ev_init(eflag, vflag);

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair==1)
    error->all(FLERR,"Pair style NEQUIP requires 'newton off'");

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = std::accumulate(numneigh, numneigh+ntotal, 0);

  torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  torch::Tensor tag2type_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  torch::Tensor cell_tensor = torch::zeros({3,3});

  auto pos = pos_tensor.accessor<float, 2>();
  long edges[2*nedges];
  float edge_cell_shifts[3*nedges];
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float,2>();

  // Inverse mapping from tag to "real" atom index
  std::vector<int> tag2i(inum);

  // Loop over real atoms to store tags, types and positions
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    // Inverse mapping from tag to x/f atom index
    tag2i[itag-1] = i; // tag is probably 1-based
    tag2type[itag-1] = type_mapper[itype];
    pos[itag-1][0] = x[i][0];
    pos[itag-1][1] = x[i][1];
    pos[itag-1][2] = x[i][2];
  }

  // Get cell
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  /*
  std::cout << "cell: " << cell_tensor << "\n";
  std::cout << "tag2i: " << "\n";
  for(int itag = 0; itag < inum; itag++){
    std::cout << tag2i[itag] << " ";
  }
  std::cout << std::endl;
  */

  auto cell_inv = cell_tensor.inverse().transpose(0,1);

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  int edge_counter = 0;
  if (debug_mode) printf("NEQUIP edges: i j xi[:] xj[:] cell_shift[:] rij\n");
  for(int ii = 0; ii < nlocal; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      // TODO: check sign
      periodic_shift[0] = x[j][0] - pos[jtag-1][0];
      periodic_shift[1] = x[j][1] - pos[jtag-1][1];
      periodic_shift[2] = x[j][2] - pos[jtag-1][2];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < cutoff*cutoff){
          torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor);
          auto cell_shift = cell_shift_tensor.accessor<float, 1>();
          float * e_vec = &edge_cell_shifts[edge_counter*3];
          e_vec[0] = std::round(cell_shift[0]);
          e_vec[1] = std::round(cell_shift[1]);
          e_vec[2] = std::round(cell_shift[2]);
          //std::cout << "cell shift: " << cell_shift_tensor << "\n";

          // TODO: double check order
          edges[edge_counter*2] = itag - 1; // tag is probably 1-based
          edges[edge_counter*2+1] = jtag - 1; // tag is probably 1-based
          edge_counter++;

          if (debug_mode){
              printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", itag-1, jtag-1,
                pos[itag-1][0],pos[itag-1][1],pos[itag-1][2],pos[jtag-1][0],pos[jtag-1][1],pos[jtag-1][2],
                e_vec[0],e_vec[1],e_vec[2],sqrt(rsq));
          }

      }
    }
  }
  if (debug_mode) printf("end NEQUIP edges\n");

  // shorten the list before sending to nequip
  torch::Tensor edges_tensor = torch::zeros({2,edge_counter}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({edge_counter,3});
  auto new_edges = edges_tensor.accessor<long, 2>();
  auto new_edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  for (int i=0; i<edge_counter; i++){

      long *e=&edges[i*2];
      new_edges[0][i] = e[0];
      new_edges[1][i] = e[1];

      float *ev = &edge_cell_shifts[i*3];
      new_edge_cell_shifts[i][0] = ev[0];
      new_edge_cell_shifts[i][1] = ev[1];
      new_edge_cell_shifts[i][2] = ev[2];
  }


  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("edge_cell_shift", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("atom_types", tag2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  if(debug_mode){
    std::cout << "NequIP model input:\n";
    std::cout << "pos:\n" << pos_tensor << "\n";
    std::cout << "edge_index:\n" << edges_tensor << "\n";
    std::cout << "edge_cell_shifts:\n" << edge_cell_shifts_tensor << "\n";
    std::cout << "cell:\n" << cell_tensor << "\n";
    std::cout << "atom_types:\n" << tag2type_tensor << "\n";
  }


  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<double, 2>();

  torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu();

  // store the total energy where LAMMPS wants it
  eng_vdwl = total_energy_tensor.data_ptr<double>()[0];

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<double, 2>();

  if(vflag){
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<double, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    virial[0] = v[0][0][0];
    virial[1] = v[0][1][1];
    virial[2] = v[0][2][2];
    virial[3] = v[0][0][1];
    virial[4] = v[0][0][2];
    virial[5] = v[0][1][2];
  }
  if(vflag_atom) {
    error->all(FLERR,"Pair style NEQUIP does not support per-atom virial");
  }

  if(debug_mode){
    std::cout << "NequIP model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
    if(vflag) std::cout << "virial: " << output.at("virial").toTensor().cpu() << std::endl;
  }

  // Write forces and per-atom energies (0-based tags here)
  for(int itag = 0; itag < inum; itag++){
    int i = tag2i[itag];
    f[i][0] = forces[itag][0];
    f[i][1] = forces[itag][1];
    f[i][2] = forces[itag][2];
    if (eflag_atom) eatom[i] = atomic_energies[itag][0];
  }


  // TODO: Performance: Depending on how the graph network works, using tags for edges may lead to shitty memory access patterns and performance.
  // It may be better to first create tag2i as a separate loop, then set edges[edge_counter][:] = (i, tag2i[jtag]).
  // Then use forces(i,0) instead of forces(itag,0).
  // Or just sort the edges somehow.

  /*
  if(device.is_cuda()){
    //torch::cuda::empty_cache();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  */
}
