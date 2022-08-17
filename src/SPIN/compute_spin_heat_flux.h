/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(spin/heat/flux,ComputeSpinHeatFlux);
// clang-format on
#else

#ifndef LMP_COMPUTE_SPIN_HEAT_FLUX_H
#define LMP_COMPUTE_SPIN_HEAT_FLUX_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSpinHeatFlux : public Compute {
 public:
  ComputeSpinHeatFlux(class LAMMPS *, int, char **);
  ~ComputeSpinHeatFlux() override;
  void init() override;
  void compute_vector() override;

 private:
  char *id_ke, *id_pe, *id_stress, *id_spin;
  class Compute *c_ke, *c_pe, *c_stress, *c_spin;

 class PairSpin **spin_pair;
 class Pair *pair;
 int npairs, npairspin;

};

}    // namespace LAMMPS_NS

#endif
#endif
