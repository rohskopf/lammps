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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(sna/grid/kk,ComputeSNAGridKokkosDevice<LMPDeviceType>);
ComputeStyle(sna/grid/kk/device,ComputeSNAGridKokkosDevice<LMPDeviceType>);
#ifdef LMP_KOKKOS_GPU
ComputeStyle(sna/grid/kk/host,ComputeSNAGridKokkosHost<LMPHostType>);
#else
ComputeStyle(sna/grid/kk/host,ComputeSNAGridKokkosDevice<LMPHostType>);
#endif
// clang-format on
#else

// clang-format off
#ifndef LMP_COMPUTE_SNA_GRID_KOKKOS_H
#define LMP_COMPUTE_SNA_GRID_KOKKOS_H

#include "compute_sna_grid.h"
#include "kokkos_type.h"
//#include "pair_snap.h"
//#include "kokkos_type.h"
//#include "neigh_list_kokkos.h"
#include "sna_kokkos.h"
//#include "pair_kokkos.h"

namespace LAMMPS_NS {

// Routines for both the CPU and GPU backend
//template<int NEIGHFLAG, int EVFLAG>
//struct TagPairSNAPComputeForce{};


// GPU backend only
/*
struct TagPairSNAPComputeNeigh{};
struct TagPairSNAPComputeCayleyKlein{};
struct TagPairSNAPPreUi{};
struct TagPairSNAPComputeUiSmall{}; // more parallelism, more divergence
struct TagPairSNAPComputeUiLarge{}; // less parallelism, no divergence
struct TagPairSNAPTransformUi{}; // re-order ulisttot from SoA to AoSoA, zero ylist
struct TagPairSNAPComputeZi{};
struct TagPairSNAPBeta{};
struct TagPairSNAPComputeBi{};
struct TagPairSNAPTransformBi{}; // re-order blist from AoSoA to AoS
struct TagPairSNAPComputeYi{};
struct TagPairSNAPComputeYiWithZlist{};
template<int dir>
struct TagPairSNAPComputeFusedDeidrjSmall{}; // more parallelism, more divergence
template<int dir>
struct TagPairSNAPComputeFusedDeidrjLarge{}; // less parallelism, no divergence
*/
struct TagComputeSNAGridLoop{};

// CPU backend only
/*
struct TagPairSNAPComputeNeighCPU{};
struct TagPairSNAPPreUiCPU{};
struct TagPairSNAPComputeUiCPU{};
struct TagPairSNAPTransformUiCPU{};
struct TagPairSNAPComputeZiCPU{};
struct TagPairSNAPBetaCPU{};
struct TagPairSNAPComputeBiCPU{};
struct TagPairSNAPZeroYiCPU{};
struct TagPairSNAPComputeYiCPU{};
struct TagPairSNAPComputeDuidrjCPU{};
struct TagPairSNAPComputeDeidrjCPU{};
*/
struct TagComputeSNAGridLoopCPU{};

//template<class DeviceType>
template<class DeviceType, typename real_type_, int vector_length_>
class ComputeSNAGridKokkos : public ComputeSNAGrid {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  static constexpr int vector_length = vector_length_;
  using real_type = real_type_;
  using complex = SNAComplex<real_type>;

  // Static team/tile sizes for device offload

#ifdef KOKKOS_ENABLE_HIP
  static constexpr int team_size_compute_neigh = 2;
  static constexpr int tile_size_compute_ck = 2;
  static constexpr int tile_size_pre_ui = 2;
  static constexpr int team_size_compute_ui = 2;
  static constexpr int tile_size_transform_ui = 2;
  static constexpr int tile_size_compute_zi = 2;
  static constexpr int tile_size_compute_bi = 2;
  static constexpr int tile_size_transform_bi = 2;
  static constexpr int tile_size_compute_yi = 2;
  static constexpr int team_size_compute_fused_deidrj = 2;
#else
  static constexpr int team_size_compute_neigh = 4;
  static constexpr int tile_size_compute_ck = 4;
  static constexpr int tile_size_pre_ui = 4;
  static constexpr int team_size_compute_ui = sizeof(real_type) == 4 ? 8 : 4;
  static constexpr int tile_size_transform_ui = 4;
  static constexpr int tile_size_compute_zi = 8;
  static constexpr int tile_size_compute_bi = 4;
  static constexpr int tile_size_transform_bi = 4;
  static constexpr int tile_size_compute_yi = 8;
  static constexpr int team_size_compute_fused_deidrj = sizeof(real_type) == 4 ? 4 : 2;
#endif

  ComputeSNAGridKokkos(class LAMMPS *, int, char **);
  ~ComputeSNAGridKokkos() override;

  void init() override;
  void compute_array() override;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeSNAGridLoop, const int& ) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeSNAGridLoopCPU, const int&) const;

 private:

  SNAKokkos<DeviceType, real_type, vector_length> snaKK;

  int chunk_size, chunk_offset;
  int host_flag;

  Kokkos::View<real_type*, DeviceType> d_radelem;              // element radii
  Kokkos::View<real_type*, DeviceType> d_wjelem;               // elements weights
  //Kokkos::View<real_type**, Kokkos::LayoutRight, DeviceType> d_coeffelem;           // element bispectrum coefficients
  Kokkos::View<real_type*, DeviceType> d_sinnerelem;           // element inner cutoff midpoint
  Kokkos::View<real_type*, DeviceType> d_dinnerelem;           // element inner cutoff half-width
  Kokkos::View<T_INT*, DeviceType> d_map;                    // mapping from atom types to elements


  typename AT::t_x_array_randomread x;
  typename AT::t_int_1d_randomread type;

};

// These wrapper classes exist to make the compute style factory happy/avoid having
// to extend the compute  style factory to support Compute classes w/an arbitrary number
// of extra template parameters

template <class DeviceType>
class ComputeSNAGridKokkosDevice : public ComputeSNAGridKokkos<DeviceType, SNAP_KOKKOS_REAL, SNAP_KOKKOS_DEVICE_VECLEN> {

 private:
  using Base = ComputeSNAGridKokkos<DeviceType, SNAP_KOKKOS_REAL, SNAP_KOKKOS_DEVICE_VECLEN>;

 public:

  ComputeSNAGridKokkosDevice(class LAMMPS *, int, char **);

  void init() override;
  void compute_array() override;

};

#ifdef LMP_KOKKOS_GPU
template <class DeviceType>
class ComputeSNAGridKokkosHost : public ComputeSNAGridKokkos<DeviceType, SNAP_KOKKOS_REAL, SNAP_KOKKOS_HOST_VECLEN> {

 private:
  using Base = ComputeSNAGridKokkos<DeviceType, SNAP_KOKKOS_REAL, SNAP_KOKKOS_HOST_VECLEN>;

 public:

  ComputeSNAGridKokkosHost(class LAMMPS *, int, char **);

  void init() override;
  void compute_array() override;

};
#endif

}

#endif
#endif

// The following will compile with the chunk in cpp file but we're gonna try wrapper like pair snap.
/*
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(sna/grid/kk,ComputeSNAGridKokkos<LMPDeviceType>);
ComputeStyle(sna/grid/kk/device,ComputeSNAGridKokkos<LMPDeviceType>);
ComputeStyle(sna/grid/kk/host,ComputeSNAGridKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_COMPUTE_SNA_GRID_KOKKOS_H
#define LMP_COMPUTE_SNA_GRID_KOKKOS_H

#include "compute_sna_grid.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

//template<int CSTYLE, int NCOL>
//struct TagComputeCoordAtom{};

template<class DeviceType>
class ComputeSNAGridKokkos : public ComputeSNAGrid {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  ComputeSNAGridKokkos(class LAMMPS *, int, char **);
  ~ComputeSNAGridKokkos() override;

 private:

};

}

#endif
#endif
*/

