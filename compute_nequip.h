/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(nequip,ComputeNEQUIP)

#else

#ifndef LMP_COMPUTE_NEQUIP_H
#define LMP_COMPUTE_NEQUIP_H

#include "compute.h"

#include <torch/torch.h>
#include <string.h>

namespace LAMMPS_NS {

class ComputeNEQUIP : public Compute {
 public:
  ComputeNEQUIP(class LAMMPS *, int, char**);
  ~ComputeNEQUIP();
  void compute_vector() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void allocate();

  double cutoff;
  torch::jit::Module model;
  torch::Device device = torch::kCPU;

 protected:
  int * type_mapper;
  class NeighList *list;
  std::string quantity;

};

}

#endif
#endif

