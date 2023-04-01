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
#include "sna_kokkos.h"
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
}

// Destructor

template<class DeviceType, typename real_type, int vector_length>
ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::~ComputeSNAGridKokkos()
{
  //if (copymode) return;

  //memoryKK->destroy_kokkos(k_eatom,eatom);
  //memoryKK->destroy_kokkos(k_vatom,vatom);
}

// Init

template<class DeviceType, typename real_type, int vector_length>
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::init()
{

  if (host_flag) {
    // The following lmp->kokkos will compile error with pointer to incomplete class type not allowed.
    //if (lmp->kokkos->nthreads > 1)
    //  error->all(FLERR,"Compute style sna/grid/kk can currently only run on a single "
    //                     "CPU thread");

    ComputeSNAGrid::init();
    return;
  }

  printf("^^^ inside compute sna grid kokkos init\n");

}

// Compute

template<class DeviceType, typename real_type, int vector_length>
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::compute_array()
{
  if (host_flag) {
    /*
    atomKK->sync(Host,X_MASK|F_MASK|TYPE_MASK);
    PairSNAP::compute(eflag_in,vflag_in);
    atomKK->modified(Host,F_MASK);
    */
    return;
  }

  printf("^^^ inside compute sna grid kokkos compute\n");
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
