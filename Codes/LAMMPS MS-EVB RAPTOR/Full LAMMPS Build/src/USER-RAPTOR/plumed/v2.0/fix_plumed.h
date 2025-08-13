
#ifdef FIX_CLASS

FixStyle(plumed,FixPlumed)

#else

#ifndef LMP_FIX_PLUMED_H
#define LMP_FIX_PLUMED_H

#include "fix.h"
// the plumed header that defines the class//
#include "../Plumed.h"

namespace LAMMPS_NS {

class FixPlumed : public Fix {
 public:
  FixPlumed(class LAMMPS *, int, char **);
  ~FixPlumed();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void pre_force(int);
  void pre_force_respa(int, int, int);
  void min_pre_force(int);

 private:
// pointer to plumed object:
  PLMD::Plumed*p;
// number of atoms local to this process:
  int nlocal;
// array of atom indexes local to this process:
  int*gatindex;
// array of masses for local atoms:
  double*masses;
// array of charges for local atoms:
  double*charges;
// this is something to enable respa
  int nlevels_respa;

  // number of atoms local to this process + RAPTOR cecs
  int nlocal_raptor;

  int me;
  
  int lmax;
  double **lx;
  double **lf;
};

};

#endif
#endif
