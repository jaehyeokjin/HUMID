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

#ifdef _RAPTOR_GPU

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm/gpu,EVB_PPPMGPU)

#else

#ifndef EVB_PPPM_GPU_H
#define EVB_PPPM_GPU_H

#include "EVB_pppm.h"

namespace LAMMPS_NS {
  
class EVB_PPPMGPU : public EVB_PPPM {
 public:
  EVB_PPPMGPU(class LAMMPS *, int, char **);
  virtual ~EVB_PPPMGPU();
  virtual void init();
  virtual void setup();
  virtual int timing_1d(int, double &);
  virtual int timing_3d(int, double &);
  virtual double memory_usage();

  virtual void particle_map_cplx();
  virtual void map2density_one_subtract(int);

  virtual void compute_env(int);
  virtual void compute_cplx(int);
  virtual void compute_exch(int);

 protected:
  FFT_SCALAR ***density_brick_gpu, ***vd_brick;
  bool kspace_split, im_real_space;
  int old_nlocal;
  double poisson_time;

  virtual void brick2fft();
  virtual void poisson_ik(int);

  // GridComm
  
  virtual void pack_forward(int, FFT_SCALAR *, int, int *);
  virtual void unpack_forward(int, FFT_SCALAR *, int, int *);
  virtual void pack_reverse(int, FFT_SCALAR *, int, int *);
  virtual void unpack_reverse(int, FFT_SCALAR *, int, int *);

  FFT_SCALAR ***create_3d_offset(int, int, int, int, int, int, const char *,
                                 FFT_SCALAR *, int);
  void destroy_3d_offset(FFT_SCALAR ***, int, int);
};

}

#endif
#endif
#endif
