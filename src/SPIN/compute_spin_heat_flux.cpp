// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: German Samolyuk (ORNL) and
                         Mario Pinto (Computational Research Lab, Pune, India)
------------------------------------------------------------------------- */

#include "compute_spin_heat_flux.h"

#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "force.h"
#include "error.h"
#include "memory.h"
#include "pair_spin.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair_hybrid.h"

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeSpinHeatFlux::ComputeSpinHeatFlux(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  id_ke(nullptr), id_pe(nullptr), id_stress(nullptr), spin_pair(nullptr)
{
  if (narg != 7) error->all(FLERR,"Illegal compute heat/flux command");

  vector_flag = 1;
  size_vector = 6;
  extvector = 1;

  // store ke/atom, pe/atom, stress/atom IDs used by heat flux computation
  // insure they are valid for these computations

  id_ke = utils::strdup(arg[3]);
  id_pe = utils::strdup(arg[4]);
  id_stress = utils::strdup(arg[5]);
  id_spin = utils::strdup(arg[6]);

  int ike = modify->find_compute(id_ke);
  int ipe = modify->find_compute(id_pe);
  int istress = modify->find_compute(id_stress);
  int ispin = modify->find_compute(id_spin);
  if (ike < 0 || ipe < 0 || istress < 0)
    error->all(FLERR,"Could not find compute heat/flux compute ID");
  if (strcmp(modify->compute[ike]->style,"ke/atom") != 0)
    error->all(FLERR,"Compute heat/flux compute ID does not compute ke/atom");
  if (modify->compute[ipe]->peatomflag == 0)
    error->all(FLERR,"Compute heat/flux compute ID does not compute pe/atom");
  if (modify->compute[istress]->pressatomflag != 1
      && modify->compute[istress]->pressatomflag != 2)
    error->all(FLERR,
               "Compute heat/flux compute ID does not compute stress/atom or centroid/stress/atom");
  //if (modify->compute[ispin]->pair_spin_flag != 1)
  //  error->all(FLERR, "Compute spin/heat/flux compute ID does not compute pairwise spin interactions");

  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputeSpinHeatFlux::~ComputeSpinHeatFlux()
{
  delete [] id_ke;
  delete [] id_pe;
  delete [] id_stress;
  delete [] id_spin;
  delete [] vector;
  delete [] spin_pair;
}

/* ---------------------------------------------------------------------- */

void ComputeSpinHeatFlux::init()
{
  // error checks

  int ike = modify->find_compute(id_ke);
  int ipe = modify->find_compute(id_pe);
  int istress = modify->find_compute(id_stress);
  int ispin = modify->find_compute(id_spin);
  if (ike < 0 || ipe < 0 || istress < 0 || ispin < 0)
    error->all(FLERR,"Could not find compute heat/flux compute ID");

  c_ke = modify->compute[ike];
  c_pe = modify->compute[ipe];
  c_stress = modify->compute[istress];
  c_spin = modify->compute[ispin];

 int nlocal = atom->nlocal;
 //int iptm = modify->find_compute(id_ptm);
 //int ivoro = modify->find_compute(id_voro);
 //int istress = modify->find_compute(id_stress);
 //if (iptm < 0 || istress < 0 || ivoro < 0)
 // error->all(FLERR,"Could not find one or more of provided compute IDs");

 //c_voro = modify->compute[ivoro];
 //c_ptm = modify->compute[iptm];
 //c_stress = modify->compute[istress];
 npairspin = 0;
 PairHybrid *hybrid = (PairHybrid *)force->pair_match("^hybrid",0);
 if (force->pair_match("^spin",0,0)) {    // only one Pair/Spin style
  pair = force->pair_match("^spin",0,0);
  if (hybrid == nullptr) npairs = 1;
  else npairs = hybrid->nstyles;
  npairspin = 1;
 } else if (force->pair_match("^spin",0,1)) { // more than one Pair/Spin style
  pair = force->pair_match("^spin",0,1);
  if (hybrid == nullptr) npairs = 1;
  else npairs = hybrid->nstyles;
  for (int i = 0; i<npairs; i++) {
   if (force->pair_match("^spin",0,i)) {
    npairspin ++;
   }
  }
 }

 if (npairspin > 0) {
  spin_pair = new PairSpin*[npairspin];
 }

 int count1 = 0;
 if (npairspin == 1) {
  count1 = 1;
  spin_pair[0] = (PairSpin *) force->pair_match("^spin",0,0);
 } else if (npairspin > 1) {
  for (int i = 0; i<npairs; i++) {
    if (force->pair_match("^spin",0,i)) {
      spin_pair[count1] = (PairSpin *) force->pair_match("^spin",0,i);
      count1++;
    }
  }
 }

}

/* ---------------------------------------------------------------------- */

void ComputeSpinHeatFlux::compute_vector()
{
  invoked_vector = update->ntimestep;

  // invoke 4 computes if they haven't been already

  if (!(c_ke->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_ke->compute_peratom();
    c_ke->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  if (!(c_pe->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_pe->compute_peratom();
    c_pe->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  if (!(c_stress->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_stress->compute_peratom();
    c_stress->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  if (!(c_spin->invoked_flag & Compute::INVOKED_PERATOM)) {
    c_spin->compute_peratom();
    c_spin->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // heat flux vector = jc[3] + jv[3]
  // jc[3] = convective portion of heat flux = sum_i (ke_i + pe_i) v_i[3]
  // jv[3] = virial portion of heat flux = sum_i (stress_tensor_i . v_i[3])
  // normalization by volume is not included

  double *ke = c_ke->vector_atom;
  double *pe = c_pe->vector_atom;
  double **stress = c_stress->array_atom;
  int *type = atom->type;
  double **x = atom->x;
  double **sp = atom->sp;
  double **fm = atom->fm;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double jc[3] = {0.0,0.0,0.0};
  double jv[3] = {0.0,0.0,0.0};
  double eng;

  //printf("----- compute_vector()\n");
  //printf("%f %f %f\n", sp[5][0], sp[5][1], sp[5][2]);
  //printf("%d\n", atom->natoms);

  // calculate ds/dt = f cross s

  double **dsdt;
  memory->create(dsdt, atom->natoms, 3, "compute:dsdt");

  int i,j,ii,jj,itype,jtype,inum,jnum; 
  int *ilist,*jlist,*numneigh,**firstneigh;

  inum = spin_pair[0]->list->inum;
  ilist = spin_pair[0]->list->ilist;
  numneigh = spin_pair[0]->list->numneigh;
  firstneigh = spin_pair[0]->list->firstneigh;
  //printf("inum: %d\n", inum);

  double spi[3]; // = {sp[0][0], sp[0][1], sp[0][2]};
  double spj[3]; // = {sp[1][0], sp[1][1], sp[1][2]};
  //printf("%f %f\n", x[0][0], x[10][0]);
  //double rsq = (x[0][0]-x[10][0])*(x[0][0]-x[10][0]);
  //double test = spin_pair[0]->compute_exchange_pair(0,1,rsq,spi,spj);
  //printf("test: %f\n", test);


  double crossproduct[3];
  double dotproduct;
  double delx, dely, delz, rsq;
  double spin_interaction;
  double heatflux[3];
  heatflux[0] = 0;
  heatflux[1] = 0;
  heatflux[2] = 0;
  for (ii=0; ii < nlocal; ii++){

    i = ilist[ii];
    itype = type[i];

    // calculate dsi/dt = fi X si

    crossproduct[0] =       atom->fm[i][1]*atom->sp[i][2] - atom->fm[i][2]*atom->sp[i][1];
    crossproduct[1] = -1.0*(atom->fm[i][0]*atom->sp[i][2] - atom->fm[i][2]*atom->sp[i][0]);
    crossproduct[2] =       atom->fm[i][0]*atom->sp[i][1] - atom->fm[i][1]*atom->sp[i][0];

    // loop over neighbors

    jnum = numneigh[i];
    jlist = firstneigh[i];
    for (jj=0; jj < jnum; jj++){
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = x[i][0]-x[j][0];
      dely = x[i][1]-x[j][1];
      delz = x[i][2]-x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      spi[0] = sp[i][0];
      spi[1] = sp[i][1];
      spi[2] = sp[i][2];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];

      spin_interaction = spin_pair[0]->compute_exchange_pair(i,j,rsq,spi,spj);

      dotproduct = crossproduct[0]*spj[0] + crossproduct[1]*spj[1] + crossproduct[2]*spj[2];

      heatflux[0] += 0.5*spin_interaction*dotproduct*delx;
      heatflux[1] += 0.5*spin_interaction*dotproduct*dely;
      heatflux[2] += 0.5*spin_interaction*dotproduct*delz;
    }
  }

  // clean up

  memory->destroy(dsdt);

  // heat flux via centroid atomic stress
  if (c_stress->pressatomflag == 2) {
    for (int i = 0; i < nlocal; i++) {

      //printf("%f %f %f\n", sp[i][0], sp[i][1], sp[i][2]);
      if (mask[i] & groupbit) {


        eng = pe[i] + ke[i];
        jc[0] += eng*v[i][0];
        jc[1] += eng*v[i][1];
        jc[2] += eng*v[i][2];
        // stress[0]: rijx*fijx
        // stress[1]: rijy*fijy
        // stress[2]: rijz*fijz
        // stress[3]: rijx*fijy
        // stress[4]: rijx*fijz
        // stress[5]: rijy*fijz
        // stress[6]: rijy*fijx
        // stress[7]: rijz*fijx
        // stress[8]: rijz*fijy
        // jv[0]  = rijx fijx vjx + rijx fijy vjy + rijx fijz vjz
        jv[0] -= stress[i][0]*v[i][0] + stress[i][3]*v[i][1] +
          stress[i][4]*v[i][2];
        // jv[1]  = rijy fijx vjx + rijy fijy vjy + rijy fijz vjz
        jv[1] -= stress[i][6]*v[i][0] + stress[i][1]*v[i][1] +
          stress[i][5]*v[i][2];
        // jv[2]  = rijz fijx vjx + rijz fijy vjy + rijz fijz vjz
        jv[2] -= stress[i][7]*v[i][0] + stress[i][8]*v[i][1] +
          stress[i][2]*v[i][2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        eng = pe[i] + ke[i];
        jc[0] += eng*v[i][0];
        jc[1] += eng*v[i][1];
        jc[2] += eng*v[i][2];
        jv[0] -= stress[i][0]*v[i][0] + stress[i][3]*v[i][1] +
          stress[i][4]*v[i][2];
        jv[1] -= stress[i][3]*v[i][0] + stress[i][1]*v[i][1] +
          stress[i][5]*v[i][2];
        jv[2] -= stress[i][4]*v[i][0] + stress[i][5]*v[i][1] +
          stress[i][2]*v[i][2];
      }
    }
  }

  // convert jv from stress*volume to energy units via nktv2p factor

  double nktv2p = force->nktv2p;
  jv[0] /= nktv2p;
  jv[1] /= nktv2p;
  jv[2] /= nktv2p;

  // sum across all procs
  // 1st 3 terms are total heat flux
  // 2nd 3 terms are just conductive portion

  //double data[6] = {jc[0]+jv[0],jc[1]+jv[1],jc[2]+jv[2],jc[0],jc[1],jc[2]};
  double data[6] = {heatflux[0], heatflux[1], heatflux[2], jc[0], jc[1], jc[2]};
  MPI_Allreduce(data,vector,6,MPI_DOUBLE,MPI_SUM,world);
}
