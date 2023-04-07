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
#include "pair_snap_kokkos.h"

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

  k_cutsq = tdual_fparams("ComputeSNAGridKokkos::cutsq",atom->ntypes+1,atom->ntypes+1);
  auto d_cutsq = k_cutsq.template view<DeviceType>();
  rnd_cutsq = d_cutsq;

  host_flag = (execution_space == Host);

  // ComputeSNAGrid constructor allocates `map` so let's do same here.
  // actually, let's move this down to init
  //int n = atom->ntypes;
  //printf("^^^ realloc d_map\n");
  //MemKK::realloc_kokkos(d_map,"ComputeSNAGridKokkos::map",n+1);
  

  printf("^^^^^ cutsq: %f\n", cutsq[1][1]);

  cutsq_tmp = cutsq[1][1];
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

  if (DeviceType::in_parallel()) {
    printf("^^^ compute_array() is a host function\n");
  } else {
    printf("^^^ compute_array() is not a host function\n");
  }

  if (host_flag) {
    /*
    atomKK->sync(Host,X_MASK|F_MASK|TYPE_MASK);
    PairSNAP::compute(eflag_in,vflag_in);
    atomKK->modified(Host,F_MASK);
    */
    return;
  }

  copymode = 1;

  zlen = nzhi-nzlo+1;
  ylen = nyhi-nylo+1;
  xlen = nxhi-nxlo+1;
  printf("^^^ nzlo nzhi nylo nyhi nxlo nxhi: %d %d %d %d %d %d\n", nzlo, nzhi, nylo, nyhi, nxlo, nxhi);
  total_range = (nzhi-nzlo+1)*(nyhi-nylo+1)*(nxhi-nxlo+1);

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  // This will error because trying to access host view on the device:
  //printf("x(0,0): %f\n", x(0,0));
  type = atomKK->k_type.view<DeviceType>();
  k_cutsq.template sync<DeviceType>();


  MemKK::realloc_kokkos(d_ninside,"PairSNAPKokkos:ninside",total_range);

  //printf("^^^ nzlo nzhi nylo nyhi nxlo nxhi: %d %d %d %d %d %d\n", nzlo, nzhi, nylo, nyhi, nxlo, nxhi);
  
  // Pair snap/kk uses grow_ij with some max number of neighs but compute sna/grid uses total 
  // number of atoms.
  
  //const int ntotal = atomKK->nlocal + atomKK->nghost;
  ntotal = atomKK->nlocal + atomKK->nghost;
  //printf("^^^ ntotal:  %d\n", ntotal);

  // ensure rij, inside, and typej are of size jnum
  // snaKK.grow_rij(int, int) requires 2 args where one is a chunksize.

  chunk_size = MIN(chunksize, total_range); // "chunksize" variable is set by user
  //printf("^^^ chunk_size: %d\n", chunk_size);
  snaKK.grow_rij(chunk_size, ntotal);

  // Launch 3 teams of the maximum number of threads per team
  //const int team_size_max = team_policy(3, 1).team_size_max(
  //    TagCSNAGridTeamPolicy, Kokkos::ParallelForTag());
  //typename Kokkos::TeamPolicy<DeviceType, TagCSNAGridTeamPolicy> team_policy_test(3,1);

  // Using custom policy:
  /* 
  CSNAGridTeamPolicy<DeviceType, team_size_compute_neigh ,TagCSNAGridTeam> team_policy(chunk_size,team_size_compute_neigh,vector_length);
  //team_policy = team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for("TeamPolicy",team_policy,*this);
  */


  chunk_size = total_range; 
  printf("%d %d %d\n", chunk_size, team_size_compute_neigh, vector_length);
  // team_size_compute_neigh is defined in `pair_snap_kokkos.h`
  int scratch_size = scratch_size_helper<int>(team_size_compute_neigh * ntotal);
  // The following launches `chunk_size` teams, each with `team_size` threads, using a vector length of `vector_length`.
  typename Kokkos::TeamPolicy<DeviceType, TagCSNAGridTeam> policy_team(chunk_size,team_size_compute_neigh,vector_length);
  policy_team = policy_team.set_scratch_size(0, Kokkos::PerTeam(scratch_size)); 
  Kokkos::parallel_for("TeamPolicy",policy_team,*this);



  //ComputeNeigh
   
  {
    // team_size_compute_neigh is defined in `pair_snap_kokkos.h`
    //int scratch_size = scratch_size_helper<int>(team_size_compute_neigh * ntotal);

    //SnapAoSoATeamPolicy<DeviceType, team_size_compute_neigh, TagCSNAGridComputeNeigh> policy_neigh(chunk_size,team_size_compute_neigh,vector_length);
    //policy_neigh = policy_neigh.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    //Kokkos::parallel_for("ComputeNeigh",policy_neigh,*this);
  }
 

  // let's try a simple parallel for loop
  // NOTE: We get the compiler error calling host function DeviceType::in_parallel() in this 
  // function, because this is a host-device function.
  /*
  typename Kokkos::RangePolicy<DeviceType,TagComputeSNAGridLoop> policy_loop(0,4);
  Kokkos::parallel_for("Loop",policy_loop,*this);
  */


  // Simple working loop:
  /* 
  Kokkos::parallel_for("Loop1", 4, KOKKOS_LAMBDA (const int& i) {
    printf("Greeting from iteration %i\n",i);
  });
  */

  /*
  // NOTE: We get the compiler error calling host function DeviceType::in_parallel() in this 
  // function, because this is a host-device function.
  const int chunk_size_div = (chunk_size + vector_length - 1) / vector_length;
  Snap3DRangePolicy<DeviceType, tile_size_compute_ck, TagCSNAGridComputeCayleyKlein>
      policy_compute_ck({0,0,0},{vector_length,ntotal,chunk_size_div},{vector_length,tile_size_compute_ck,1});
  Kokkos::parallel_for("ComputeCayleyKlein",policy_compute_ck,*this);
  */

  // Simple example of 3D MD range policy.
  // Begin loop over grid points.
  /*
  // NOTE: We don't get the compiler error calling host function DeviceType::in_parallel() in this 
  // function, but we get it in the above function.
  int n = 3; // bounds for mdrange policy
  typename Kokkos::MDRangePolicy<DeviceType, Kokkos::IndexType<int>, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>, TagComputeSNAGrid3D> policy_3d({0,0,0},{n,n,n});
  Kokkos::parallel_for("3D",policy_3d,*this);
  */

  printf("^^^ End ComputeSNAGridKokkos compute_array()\n");
}

/* ----------------------------------------------------------------------
   Begin routines that are unique to the GPU codepath. These take advantage
   of AoSoA data layouts and scratch memory for recursive polynomials
------------------------------------------------------------------------- */

/*
 Simple team policy functor seeing how many layers deep we can go with the parallelism.
 */
template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagCSNAGridTeam,const typename Kokkos::TeamPolicy<DeviceType,TagCSNAGridTeam>::member_type& team) const {

  // this function is following the same procedure in ComputeNeigh of PairSNAPKokkos

  SNAKokkos<DeviceType, real_type, vector_length> my_sna = snaKK;

  // basic quantities associated with this team:
  // team_rank : rank of thread in this team
  // league_rank : rank of team in this league
  // team_size : number of threads in this team
  //printf("%d %d %d\n", team.team_rank(), team.league_rank(), team.team_size());

  // extract loop index
  int ii = team.team_rank() + team.league_rank() * team.team_size();
  if (ii >= chunk_size) return;

  // get a pointer to scratch memory
  // This is used to cache whether or not an atom is within the cutoff.
  // If it is, type_cache is assigned to the atom type.
  // If it's not, it's assigned to -1.
  const int tile_size = ntotal; // number of elements per thread
  const int team_rank = team.team_rank();
  const int scratch_shift = team_rank * tile_size; // offset into pointer for entire team
  //printf("ntotal scratch_shift: %d %d\n", ntotal, scratch_shift);
  int* type_cache = (int*)team.team_shmem().get_shmem(team.team_size() * tile_size * sizeof(int), 0) + scratch_shift;

  //printf("ii: %d\n", ii);

  // convert to grid indices

  int iz = ii/(xlen*ylen);
  int i2 = ii - (iz*xlen*ylen);
  int iy = i2/xlen;
  int ix = i2 % xlen;
  iz += nzlo;
  iy += nylo;
  ix += nxlo;

  double xgrid[3];
  //int igrid = iz * (nx * ny) + iy * nx + ix;

  // these end up being the same...?
  //printf("ii igrid: %d %d\n", ii, igrid);

  // grid2x converts igrid to ix,iy,iz like we've done before
  //grid2x(igrid, xgrid);
  xgrid[0] = ix * delx;
  xgrid[1] = iy * dely;
  xgrid[2] = iz * delz;
  const double xtmp = xgrid[0];
  const double ytmp = xgrid[1];
  const double ztmp = xgrid[2];

  // currently, all grid points are type 1
  // not clear what a better choice would be

  const int itype = 1;
  const int ielem = d_map[itype];
  const double radi = d_radelem[ielem];

  // We need a DomainKokkos::lamda2x parallel for loop here, but let's ignore for now.
  if (triclinic){
    printf("We are triclinic %f %f %f\n", xtmp, ytmp, ztmp);
  } else {
    //printf("We are not triclinic\n");
  }

  // can check xgrid positions with original
  //printf("%f %f %f\n", xgrid[0], xgrid[1], xgrid[2]);

  // Compute the number of neighbors, store rsq
  int ninside = 0;
  // want to loop over ntotal... keep getting seg fault when accessing type_cache[j]?
  //printf("ntotal: %d\n", ntotal);
  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,ntotal),
    [&] (const int j, int& count) {

    // From pair snap/kk :
    /*
    T_INT j = d_neighbors(i,jj);
    const F_FLOAT dx = x(j,0) - xtmp;
    const F_FLOAT dy = x(j,1) - ytmp;
    const F_FLOAT dz = x(j,2) - ztmp;
    */
    // From compute sna/grid/kk :
    /*
    const double delx = xtmp - x[j][0];
    const double dely = ytmp - x[j][1];
    const double delz = ztmp - x[j][2];
    */
    const F_FLOAT dx = x(j,0) - xtmp;
    const F_FLOAT dy = x(j,1) - ytmp;
    const F_FLOAT dz = x(j,2) - ztmp;
    //printf("dx: %f\n", dx);

    //const double rsq = delx * delx + dely * dely + delz * delz;
    int jtype = type(j);
    //printf("jtype: %d\n", jtype);
    //int jelem = 0;
    //if (rsq < cutsq[jtype][jtype] && rsq > 1e-20) {
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    //if (rsq >= rnd_cutsq(itype,jtype)) {
    if (rsq >= cutsq_tmp){
      jtype = -1; // use -1 to signal it's outside the radius
    }
    //printf("jtype rsq rnd_cutsq: %d %f %f\n", jtype, rsq, rnd_cutsq(itype, jtype));

    if (j > 340){
      printf("j: %d\n", j);
    }

    //printf("j: %d\n", j);
    type_cache[j] = jtype;

    if (jtype >= 0)
     count++;

  }, ninside);

  printf("ninside: %d\n", ninside);

  d_ninside(ii) = ninside;

  // TODO: Make sure itype is appropriate instead of ielem
  Kokkos::parallel_scan(Kokkos::ThreadVectorRange(team,ntotal),
    [&] (const int j, int& offset, bool final) {

    const int jtype = type_cache[j];

    if (jtype >= 0) {
      if (final) {
        const F_FLOAT dx = x(j,0) - xtmp;
        const F_FLOAT dy = x(j,1) - ytmp;
        const F_FLOAT dz = x(j,2) - ztmp;
        const int jelem = d_map[jtype];
        my_sna.rij(ii,offset,0) = static_cast<real_type>(dx);
        my_sna.rij(ii,offset,1) = static_cast<real_type>(dy);
        my_sna.rij(ii,offset,2) = static_cast<real_type>(dz);
        my_sna.wj(ii,offset) = static_cast<real_type>(d_wjelem[jelem]);
        my_sna.rcutij(ii,offset) = static_cast<real_type>((radi + d_radelem[jelem])*rcutfac);
        my_sna.inside(ii,offset) = j;
        if (switchinnerflag) {
          my_sna.sinnerij(ii,offset) = 0.5*(d_sinnerelem[itype] + d_sinnerelem[jelem]);
          my_sna.dinnerij(ii,offset) = 0.5*(d_dinnerelem[itype] + d_dinnerelem[jelem]);
        }
        if (chemflag)
          my_sna.element(ii,offset) = jelem;
        else
          my_sna.element(ii,offset) = 0;
      }
      offset++;
    }
  });
}

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagComputeSNAGridLoop,const int& asdf) const {
  //printf("inside parallel for\n");
  printf("%d\n", asdf);

  /*
  typename Kokkos::RangePolicy<DeviceType,TagComputeSNAGridLoop> policy_loop(0,2);
  Kokkos::parallel_for("Loop",policy_loop,*this);
  */

}

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagComputeSNAGrid3D,const int& iz, const int& iy, const int& ix) const {
  //printf("inside parallel for\n");
  printf("^^^ ix iy iz: %d %d %d\n", ix, iy, iz);

  // Let's try to do calculations inside here and see what happens.

  if (host_flag){

  } else { // GPU

#ifdef LMP_KOKKOS_GPU

    // Pre-compute ceil(chunk_size / vector_length) for code cleanliness
    const int chunk_size_div = (chunk_size + vector_length - 1) / vector_length;
    //printf("^^^ ComputeSNAGrid3D chunk_size_div: %d\n", chunk_size_div);
    //ComputeCayleyKlein
    {
      //printf("^^^ inside bracket\n");

      if (DeviceType::in_parallel()) {
        printf("operator() of TagComputeSNAGrid3D is a host function\n");
      } else {
        printf("operator() of TagComputeSNAGrid3D is not a host function\n");
      }
  
      // This returns true:
      //auto test = std::is_same<DeviceType,LMPDeviceType>::value;
      //printf("%d\n", test);

      // tile_size_compute_ck is defined in `compute_sna_grid_kokkos.h`
      // The policy constructors are host functions so this raises compiler warning and also 
      // isnt allowed... we should use teams.
      /* 
      Snap3DRangePolicy<DeviceType, tile_size_compute_ck, TagCSNAGridComputeCayleyKlein>
          policy_compute_ck({0,0,0},{vector_length,ntotal,chunk_size_div},{vector_length,tile_size_compute_ck,1});
      printf("^^^ begin parallel_for\n");
      Kokkos::parallel_for("ComputeCayleyKlein",policy_compute_ck,*this);
      */
      
    }
#endif // LMP_KOKKOS_GPU

  }
  

}

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagCSNAGridComputeCayleyKlein,const int iatom_mod, const int jnbor, const int iatom_div) const {
  SNAKokkos<DeviceType, real_type, vector_length> my_sna = snaKK;

  printf("^^^ ComputeCayleyKlein\n");

  /*
  if (DeviceType::in_parallel()) {
    printf("operator() of TagCSNAGridComputeCayleyKlein is a host function\n");
  } else {
    printf("operator() of TagCSNAGridComputeCayleyKlein is not a host function\n");
  }
  */

  const int ii = iatom_mod + iatom_div * vector_length;
  if (ii >= chunk_size) return;

  const int ninside = ntotal; //d_ninside(ii);
  if (jnbor >= ninside) return;

  my_sna.compute_cayley_klein(iatom_mod,jnbor,iatom_div);
}

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagCSNAGridPreUi, const int iatom_mod, const int j, const int iatom_div) const {
  SNAKokkos<DeviceType, real_type, vector_length> my_sna = snaKK;

  const int ii = iatom_mod + iatom_div * vector_length;
  if (ii >= chunk_size) return;

  int itype = type(ii);
  int ielem = d_map[itype];

  my_sna.pre_ui(iatom_mod, j, ielem, iatom_div);
}

template<class DeviceType, typename real_type, int vector_length>
KOKKOS_INLINE_FUNCTION
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::operator() (TagCSNAGridComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType,TagCSNAGridComputeNeigh>::member_type& team) const {


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
   utility functions
------------------------------------------------------------------------- */

template<class DeviceType, typename real_type, int vector_length>
template<class TagStyle>
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::check_team_size_for(int inum, int &team_size) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelForTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

template<class DeviceType, typename real_type, int vector_length>
template<class TagStyle>
void ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::check_team_size_reduce(int inum, int &team_size) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelReduceTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

template<class DeviceType, typename real_type, int vector_length>
template<typename scratch_type>
int ComputeSNAGridKokkos<DeviceType, real_type, vector_length>::scratch_size_helper(int values_per_team) {
  typedef Kokkos::View<scratch_type*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}

/* ---------------------------------------------------------------------- */

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
