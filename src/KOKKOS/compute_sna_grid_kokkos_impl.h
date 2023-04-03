// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Christian Trott (SNL), Stan Moore (SNL),
                         Evan Weinberg (NVIDIA)
------------------------------------------------------------------------- */

#include "compute_sna_grid_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "memory_kokkos.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
//#include "sna_kokkos.h"
#include "sna.h"
#include "update.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#define MAXLINE 1024
#define MAXWORD 3

namespace LAMMPS_NS {

// Constructor

template<class DeviceType, typename real_type, int vector_length>
ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::ComputeSNAGridKokkos(LAMMPS *lmp, int narg, char **arg) : ComputeSNAGrid(lmp, narg, arg)
{
  //respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  //k_cutsq = tdual_fparams("PairSNAPKokkos::cutsq",atom->ntypes+1,atom->ntypes+1);
  //auto d_cutsq = k_cutsq.template view<DeviceType>();
  //rnd_cutsq = d_cutsq;

  host_flag = (execution_space == Host);

  // ComputeSNAGrid constructor allocates `map` so let's do same here.
  // actually, let's move this down to init
  //int n = atom->ntypes;
  //printf("^^^ realloc d_map\n");
  //MemKK::realloc_kokkos(d_map,"ComputeSNAGridKokkos::map",n+1);
}

// Destructor

template<class DeviceType, typename real_type, int vector_length>
ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::~ComputeSNAGridKokkos()
{
  if (copymode) return;

  //memoryKK->destroy_kokkos(k_eatom,eatom);
  //memoryKK->destroy_kokkos(k_vatom,vatom);
  printf("^^^ Finish ComputeSNAGridKokkos destructor\n");
}

// Init

template<class DeviceType, typename real_type, int vector_length>
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::init()
{
  printf("^^^ Begin ComputeSNAGridKokkos init()\n");
  // The part of pair_snap_kokkos_impl.h that allocates snap params is coeff(), and it 
  // calls the original coeff function. So let's do that here: 

  ComputeSNAGrid::init();

  // Set up element lists
  printf("^^^ Begin kokkos reallocs\n");
  MemKK::realloc_kokkos(d_radelem,"ComputeSNAGridKokkos::radelem",nelements);
  MemKK::realloc_kokkos(d_wjelem,"pair:wjelem",nelements);
  // pair snap kokkos uses `ncoeffall` in the following, inherits from original.
  //MemKK::realloc_kokkos(d_coeffelem,"pair:coeffelem",nelements,ncoeff);
  MemKK::realloc_kokkos(d_sinnerelem,"pair:sinnerelem",nelements);
  MemKK::realloc_kokkos(d_dinnerelem,"pair:dinnerelem",nelements);
  int n = atom->ntypes;
  //printf("^^^ realloc d_map\n");
  printf("^^^ n: %d\n", n);
  MemKK::realloc_kokkos(d_map,"ComputeSNAGridKokkos::map",n+1);

  printf("^^^ begin mirrow view creation\n");
  auto h_radelem = Kokkos::create_mirror_view(d_radelem);
  auto h_wjelem = Kokkos::create_mirror_view(d_wjelem);
  //auto h_coeffelem = Kokkos::create_mirror_view(d_coeffelem);
  auto h_sinnerelem = Kokkos::create_mirror_view(d_sinnerelem);
  auto h_dinnerelem = Kokkos::create_mirror_view(d_dinnerelem);
  auto h_map = Kokkos::create_mirror_view(d_map);

  printf("^^^ begin loop over elements, nelements = %d\n", nelements);
  for (int ielem = 0; ielem < nelements; ielem++) {
    printf("^^^^^ ielem %d\n", ielem);
    h_radelem(ielem) = radelem[ielem];
    printf("^^^^^ 1\n");
    h_wjelem(ielem) = wjelem[ielem];
    printf("^^^^^ 2\n");
    if (switchinnerflag){
      h_sinnerelem(ielem) = sinnerelem[ielem];
      h_dinnerelem(ielem) = dinnerelem[ielem];
    }
    // pair snap kokkos uses `ncoeffall` in the following.
    //for (int jcoeff = 0; jcoeff < ncoeff; jcoeff++) {
    //  h_coeffelem(ielem,jcoeff) = coeffelem[ielem][jcoeff];
    //}
  }

  printf("^^^ begin loop over map\n");
  // NOTE: At this point it's becoming obvious that compute sna grid is not like pair snap, where 
  // some things like `map` get allocated regardless of chem flag.
  if (chemflag){ 
    for (int i = 1; i <= atom->ntypes; i++) {
      h_map(i) = map[i];
      printf("%d\n", map[i]);
    }
  }

  Kokkos::deep_copy(d_radelem,h_radelem);
  Kokkos::deep_copy(d_wjelem,h_wjelem);
  if (switchinnerflag){
    Kokkos::deep_copy(d_sinnerelem,h_sinnerelem);
    Kokkos::deep_copy(d_dinnerelem,h_dinnerelem);
  }
  if (chemflag){
    Kokkos::deep_copy(d_map,h_map);
  }

  snaKK = SNAKokkos<DeviceType, real_type, vector_length>(rfac0,twojmax,
    rmin0,switchflag,bzeroflag,chemflag,bnormflag,wselfallflag,nelements,switchinnerflag);
  snaKK.grow_rij(0,0);
  snaKK.init();

  if (host_flag) {

    // The following lmp->kokkos will compile error with pointer to incomplete class type not allowed.
    //if (lmp->kokkos->nthreads > 1)
    //  error->all(FLERR,"Compute style sna/grid/kk can currently only run on a single "
    //                     "CPU thread");

    ComputeSNAGrid::init();
    return;
  }

  printf("^^^ Finished ComputeSNAGridKokkos init\n");

}

// Compute

template<class DeviceType, typename real_type, int vector_length>
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::compute_array()
{
  printf("^^^ Begin ComputeSNAGridKokkos compute_array()\n");
  if (host_flag) {
    /*
    atomKK->sync(Host,X_MASK|F_MASK|TYPE_MASK);
    PairSNAP::compute(eflag_in,vflag_in);
    atomKK->modified(Host,F_MASK);
    */
    return;
  }

  copymode = 1;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  // This will error because trying to access host view on the device:
  //printf("x(0,0): %f\n", x(0,0));
  type = atomKK->k_type.view<DeviceType>();
  
  // Pair snap/kk uses grow_ij with some max number of neighs but compute sna/grid uses total 
  // number of atoms.
  
  const int ntotal = atomKK->nlocal + atomKK->nghost;
  printf("^^^ ntotal:  %d\n", ntotal);

  // ensure rij, inside, and typej are of size jnum
  // snaKK.grow_rij(int, int) requires 2 args where one is a chunksize.

  chunk_size = MIN(chunksize, ntotal); // "chunksize" variable is set by user
  printf("^^^ chunk_size: %d\n", chunk_size);
  snaKK.grow_rij(chunk_size, ntotal);

  // begin triple loop over grid points
  
  // experiment with MD range policy first?
  

  // let's try a simple parallel for loop
  typename Kokkos::RangePolicy<DeviceType,TagComputeSNAGridLoop> policy_loop(0,4);
  // perhaps the `this` is causing the seg fault when running from python?
  // TODO: Don't use *this here...?
  // No... that just allows to find functor
  Kokkos::parallel_for("Loop",policy_loop,*this);
  // Simple working loop:
  /* 
  Kokkos::parallel_for("Loop1", 4, KOKKOS_LAMBDA (const int& i) {
    printf("Greeting from iteration %i\n",i);
  });
  */

  printf("^^^ End ComputeSNAGridKokkos compute_array()\n");
}

/* ----------------------------------------------------------------------
   Begin routines that are unique to the GPU codepath. These take advantage
   of AoSoA data layouts and scratch memory for recursive polynomials
------------------------------------------------------------------------- */

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagComputeSNAGridLoop,const int& asdf) const {
  //printf("inside parallel for\n");
  printf("%d\n", asdf);

}

/* ----------------------------------------------------------------------
   Begin routines that are unique to the CPU codepath. These do not take
   advantage of AoSoA data layouts, but that could be a good point of
   future optimization and unification with the above kernels. It's unlikely
   that scratch memory optimizations will ever be useful for the CPU due to
   different arithmetic intensity requirements for the CPU vs GPU.
------------------------------------------------------------------------- */

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagComputeSNAGridLoopCPU,const int& ii) const {

}

/* ----------------------------------------------------------------------
   routines used by template reference classes
------------------------------------------------------------------------- */


template<class DeviceType>
ComputeSNAGridKokkosDevice<DeviceType>::ComputeSNAGridKokkosDevice(class LAMMPS *lmp, int narg, char **arg)
   : ComputeSNAGridKokkos<DeviceType, SNAP_KOKKOS_REAL, SNAP_KOKKOS_DEVICE_VECLEN>(lmp, narg, arg) { ; }

template<class DeviceType>
void ComputeSNAGridKokkosDevice<DeviceType>::init()
{
  Base::init();
}

template<class DeviceType>
void ComputeSNAGridKokkosDevice<DeviceType>::compute_array()
{
  Base::compute_array();
}

#ifdef LMP_KOKKOS_GPU
template<class DeviceType>
ComputeSNAGridKokkosHost<DeviceType>::ComputeSNAGridKokkosHost(class LAMMPS *lmp, int narg, char **arg)
   : ComputeSNAGridKokkos<DeviceType, SNAP_KOKKOS_REAL, SNAP_KOKKOS_HOST_VECLEN>(lmp, narg, arg) { ; }

template<class DeviceType>
void ComputeSNAGridKokkosHost<DeviceType>::init()
{
  Base::init();
}

template<class DeviceType>
void ComputeSNAGridKokkosHost<DeviceType>::compute_array()
{
  Base::compute_array();
}
#endif

}
