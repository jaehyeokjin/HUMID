
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#if defined (_OPENMP)

#ifdef PAIR_CLASS

PairStyle(electrode/omp,PairElectrodeOMP)

#else

#ifndef LMP_PAIR_ELECTRODE_OMP_H
#define LMP_PAIR_ELECTRODE_OMP_H

#define RAPTOR_SUPPORT

#include "pair_electrode.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

  class PairElectrodeOMP : public PairElectrode, public ThrOMP {

 public:
 
  PairElectrodeOMP(class LAMMPS *);
  virtual ~PairElectrodeOMP();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  
 /* ----- For electrode model ----- */

 public:

  /* RAPTOR support */
  
  double compute_exch(int);
};

}

#endif
#endif

#endif
