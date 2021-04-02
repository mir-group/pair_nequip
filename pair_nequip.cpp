// TODO: header file


void PairNEQUIP::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NEQUIP requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // TODO: probably also
  neighbor->requests[irequest]->ghost = 1;

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR,"Pair style Stillinger-Weber requires newton pair off");
}

void PairNEQUIP::coeff(int narg, char **arg){
  // TODO: Read and set up model.
}
// TODO: Maybe more required functions, like init_one(i,j), constructor, destructor, etc.

// Force and energy computation
void PairNEQUIP::compute(int eflag, int vflag){
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
  // Should probably be off.
  int newton_pair = force->newton_pair;

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
  // (I think this is how C++ works)
  int nedges = std::accumulate(numneigh, numneigh+ntotal, 0);

  // TODO: Torchify!
  int edges[nedges,2];
  double edgedisplacements[nedges,3];
  // TODO: do you need species/types?
  int edgetypes[nedges,2];

  // Inverse mapping from tag to "real" atom index
  std::vector<int> tag2i(inum);

  // Loop over atoms and neighbors,
  // store edges and displacements
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  int edge_counter = 0;
  for(int ii = 0; ii < ntotal; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    // Inverse mapping from tag to x/f atom index
    tag2i[itag-1] = i; // tag is probably 1-based

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      // TODO: check sign
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      edges[edge_counter][0] = itag - 1; // tag is probably 1-based
      edges[edge_counter][1] = jtag - 1; // tag is probably 1-based

      edgedisplacements[edge_counter][0] = dx;
      edgedisplacements[edge_counter][1] = dy;
      edgedisplacements[edge_counter][2] = dz;

      edgetypes[edge_counter][0] = itype - 1; // type is 1-based
      edgetypes[edge_counter][1] = jtype - 1; // type is 1-based

      edge_counter++;
    }
  }

  // TODO: de-torchify
  // Assuming forces will be (inum,3)
  auto forces = model.predict(edges, edgedisplacements, edgetypes);

  // Write forces (0-based tags here)
  for(int itag = 0; itag < inum; itag++){
    int i = tag2i[itag];
    f[i][0] = forces(itag,0);
    f[i][1] = forces(itag,1);
    f[i][2] = forces(itag,2);
  }

  // TODO: Set evdwl somehow
  // TODO: Virial stuff? (If there even is a pairwise force concept here)

  // TODO: Performance: Depending on how the graph network works, using tags for edges may lead to shitty memory access patterns and performance.
  // It may be better to first create tag2i as a separate loop, then set edges[edge_counter][:] = (i, tag2i[jtag]).
  // Then use forces(i,0) instead of forces(itag,0).
  // Or just sort the edges somehow.
}
