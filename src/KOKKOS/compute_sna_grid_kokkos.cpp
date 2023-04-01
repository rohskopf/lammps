// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_sna_grid_kokkos.h"
#include "compute_sna_grid_kokkos_impl.h"

namespace LAMMPS_NS {

template class ComputeSNAGridKokkosDevice<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeSNAGridKokkosHost<LMPHostType>;
#endif

}




// The following chunk will compile but we're gonna try a wrapper approach like pair snap.
/*
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

using namespace LAMMPS_NS;

// ----------------------------------------------------------------------

template<class DeviceType>
ComputeSNAGridKokkos<DeviceType>::ComputeSNAGridKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputeSNAGrid(lmp, narg, arg)
{

  printf("^^^ inside ComputeSNAGridKokkos constructor\n");
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

}

// ----------------------------------------------------------------------

template<class DeviceType>
ComputeSNAGridKokkos<DeviceType>::~ComputeSNAGridKokkos()
{
  if (copymode) return;


}

namespace LAMMPS_NS {
template class ComputeSNAGridKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeSNAGridKokkos<LMPHostType>;
#endif
}
*/

