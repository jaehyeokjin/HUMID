/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   Written by Chris Knight
   Based on pppm_electrode written by Yuxing Peng 
   
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm/electrode,EVB_PPPMELECTRODE)

#else

#ifndef EVB_PPPM_ELECTRODE_H
#define EVB_PPPM_ELECTRODE_H

#include "pppm_electrode.h"

namespace LAMMPS_NS {
  
class EVB_PPPMELECTRODE : public PPPM_Electrode {
 public:
    EVB_PPPMELECTRODE(class LAMMPS *, int, char **);
    virtual ~EVB_PPPMELECTRODE();

    int nlocal;
    double ** x;
    double ** f;
    double * q;

    double dipole_env; // z-component of dipole of environment for slab correction
    double dipole_r2_env;

    double ** part2grid_dr;

    FFT_SCALAR *** env_density_brick;

    int nx, ny, nz;
    int mx, my, mz;
    FFT_SCALAR x0, y0, z0;

    double qqrd2e;

    void compute(int, int) {} ;
    void compute_env(int);
    void compute_env_density(int);
    void compute_cplx(int);
    virtual void compute_cplx_eff(int); // Compute forces on cplx atoms in effective field of ENV

    void init();
    void allocate();
    void deallocate();

    void clear_density();
    void load_env_density();
    void evb_setup();
    void make_rho();
    void map2density_one(int);
    void map2density_one(int, int);
    void map2density_one_subtract(int);

    virtual void poisson_energy(int);
    void poisson(int);
    void poisson_ik(int);

    void fieldforce();
    void fieldforce_ik();
    virtual void field2force_one_ik(int, bool);
    virtual void field2force_one_ad(int, bool);

    // Used to calculate forces on only ENV atoms when using HF PPPM forces
    virtual void fieldforce_env();
    virtual void fieldforce_env_ik();

    void reduce_ev(int, bool);
    
    void slabcorr_cplx();

    void sci_compute_env(int);
    void sci_compute_cplx(int);
    void sci_compute_exch(int);
    void sci_compute_eff(int);
    void sci_compute_eff_cplx(int);
    void sci_compute_eff_mp(int);
    void sci_compute_eff_cplx_mp(int);
    void poisson_mp(int, int, int);
    void poisson_ik_mp(int, int, int);

    void slabcorr_sci_cplx();
    void slabcorr_sci_eff();

    void sci_setup_iteration();
    
    // Eliminate redundant FFTs each MD step
    int sci_first_iteration_test;           // Whether first iteration computed by partition or not.
    int * do_sci_compute_cplx_other;        // Whether to compute self-energy of other complexes in sci_compute_cplx
    double * energy_sci_compute_cplx_other; // self-energy of other complexes in sci_compute_cplx
    
    int * do_sci_compute_cplx_self;        // Whether to compute self-energy in sci_compute_cplx
    double * energy_sci_compute_cplx_self; // self-energy in sci_compute_cplx
};

}

#endif
#endif
