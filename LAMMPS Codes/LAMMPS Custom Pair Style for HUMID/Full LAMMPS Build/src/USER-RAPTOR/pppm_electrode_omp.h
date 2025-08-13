/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef KSPACE_CLASS

KSpaceStyle(pppm/electrode/omp,PPPM_ElectrodeOMP)

#else

#ifndef LMP_PPPM_Electrode_OMP_H
#define LMP_PPPM_Electrode_OMP_H

#include "lmptype.h"
#include "mpi.h"

#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define MPI_FFT_SCALAR MPI_FLOAT
#else
typedef double FFT_SCALAR;
#define MPI_FFT_SCALAR MPI_DOUBLE
#endif

#include "EVB_kspace.h"
#include "pppm_electrode.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PPPM_ElectrodeOMP : public PPPM_Electrode, public ThrOMP {
 public:
  PPPM_ElectrodeOMP(class LAMMPS *, int, char **);
  virtual ~PPPM_ElectrodeOMP();
  virtual void init();
  virtual void setup();
  virtual void compute(int, int);

 protected:
  
  void set_grid_global();
  void set_grid_local();
  void adjust_gewald();
  double newton_raphson_f();
  double derivf();
  double final_accuracy();

  virtual void allocate();
  virtual void deallocate();
  double compute_df_kspace();
  virtual double compute_qopt();
  virtual void compute_gf_ik();
  virtual void compute_gf_ad();
  
  virtual void particle_map();
  virtual void make_rho();

  virtual void poisson_ik();
  virtual void poisson_ad();

  virtual void fieldforce_ik();
  virtual void fieldforce_ad();

  void compute_rho1d_thr(FFT_SCALAR * const * const, const FFT_SCALAR &,
			 const FFT_SCALAR &, const FFT_SCALAR &);
  void compute_drho1d_thr(FFT_SCALAR * const * const, const FFT_SCALAR &,
			  const FFT_SCALAR &, const FFT_SCALAR &);
  void slabcorr();

  // triclinic

  int triclinic;               // domain settings, orthog or triclinic
  void setup_triclinic();

  /* Electrode Model */

  class PairElectrodeOMP *pair_et_omp;
  
  /* End */
  
  /* Raptor Support */
  
  virtual void compute_cplx(int);
  virtual void compute_exch(int);
  
  void evb_setup() { energy = 0.0; }
  virtual void compute_env(int) { energy = 0.0; };
  virtual void compute_env_density(int) { energy = 0.0; }
  virtual void compute_eff(int) { energy = 0.0; }
  
  void sci_setup_iteration() {};
  void sci_setup_init() {};
  void sci_compute_env(int) {};
  void sci_compute_cplx(int) {};
  void sci_compute_exch(int) {};
  void sci_compute_eff(int) {};
  void sci_compute_eff_mp(int) {};
  
};

}

#endif
#endif

#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot (yet) use PPPM_Electrode with triclinic box and kspace_modify diff ad

This feature is not yet supported.

E: Cannot (yet) use PPPM_Electrode with triclinic box and slab correction

This feature is not yet supported.

E: Cannot use PPPM_Electrode with 2d simulation

The kspace style pppm cannot be used in 2d simulations.  You can use
2d PPPM_Electrode in a 3d simulation; see the kspace_modify command.

E: Kspace style requires atom attribute q

The atom style defined does not have these attributes.

E: Cannot use nonperiodic boundaries with PPPM_Electrode

For kspace style pppm, all 3 dimensions must have periodic boundaries
unless you use the kspace_modify command to define a 2d slab with a
non-periodic z dimension.

E: Incorrect boundaries with slab PPPM_Electrode

Must have periodic x,y dimensions and non-periodic z dimension to use
2d slab option with PPPM_Electrode.

E: PPPM_Electrode order cannot be < 2 or > than %d

This is a limitation of the PPPM_Electrode implementation in LAMMPS.

E: KSpace style is incompatible with Pair style

Setting a kspace style requires that a pair style with a long-range
Coulombic or dispersion component be used.

E: Bond and angle potentials must be defined for TIP4P

Cannot use TIP4P pair potential unless bond and angle potentials
are defined.

E: Bad TIP4P angle type for PPPM_Electrode/TIP4P

Specified angle type is not valid.

E: Bad TIP4P bond type for PPPM_Electrode/TIP4P

Specified bond type is not valid.

E: Cannot (yet) use PPPM_Electrode with triclinic box and TIP4P

This feature is not yet supported.

E: Cannot use kspace solver on system with no charge

No atoms in system have a non-zero charge.

W: System is not charge neutral, net charge = %g

The total charge on all atoms on the system is not 0.0, which
is not valid for the long-range Coulombic solvers.

W: Reducing PPPM_Electrode order b/c stencil extends beyond nearest neighbor processor

This may lead to a larger grid than desired.  See the kspace_modify overlap
command to prevent changing of the PPPM_Electrode order.

E: PPPM_Electrode order < minimum allowed order

The default minimum order is 2.  This can be reset by the
kspace_modify minorder command.

E: PPPM_Electrode grid stencil extends beyond nearest neighbor processor

This is not allowed if the kspace_modify overlap setting is no.

E: KSpace accuracy must be > 0

The kspace accuracy designated in the input must be greater than zero.

E: Could not compute grid size

The code is unable to compute a grid size consistent with the desired
accuracy.  This error should not occur for typical problems.  Please
send an email to the developers.

E: PPPM_Electrode grid is too large

The global PPPM_Electrode grid is larger than OFFSET in one or more dimensions.
OFFSET is currently set to 4096.  You likely need to decrease the
requested accuracy.

E: Could not compute g_ewald

The Newton-Raphson solver failed to converge to a good value for
g_ewald.  This error should not occur for typical problems.  Please
send an email to the developers.

E: Out of range atoms - cannot compute PPPM_Electrode

One or more atoms are attempting to map their charge to a PPPM_Electrode grid
point that is not owned by a processor.  This is likely for one of two
reasons, both of them bad.  First, it may mean that an atom near the
boundary of a processor's sub-domain has moved more than 1/2 the
"neighbor skin distance"_neighbor.html without neighbor lists being
rebuilt and atoms being migrated to new processors.  This also means
you may be missing pairwise interactions that need to be computed.
The solution is to change the re-neighboring criteria via the
"neigh_modify"_neigh_modify command.  The safest settings are "delay 0
every 1 check yes".  Second, it may mean that an atom has moved far
outside a processor's sub-domain or even the entire simulation box.
This indicates bad physics, e.g. due to highly overlapping atoms, too
large a timestep, etc.

E: Cannot (yet) use K-space slab correction with compute group/group
for triclinic systems

This option is not yet supported.

E: Cannot (yet) use kspace_modify diff ad with compute group/group

This option is not yet supported.

*/
