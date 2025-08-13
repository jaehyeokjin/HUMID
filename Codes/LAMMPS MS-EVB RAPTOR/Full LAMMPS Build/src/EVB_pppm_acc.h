/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   splitted for MS-EVB by Yuxing Peng
   
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm/acc,EVB_PPPMACC)

#else

#ifndef EVB_PPPM_ACC_H
#define EVB_PPPM_ACC_H

#include "EVB_pppm.h"

namespace LAMMPS_NS {

class EVB_PPPMACC : public EVB_PPPM {
 public:
    EVB_PPPMACC(class LAMMPS *, int, char **);
    ~EVB_PPPMACC();
 
  public:
    void init();
    void setup();
    void evb_setup();
    double compute_df_kspace_acc();
    double estimate_ik_error_acc(double, double, bigint);
    double compute_qopt_acc();
    void allocate_acc();
    void deallocate_acc();
    void set_grid_global_acc();
    void set_grid_local_acc();
    void adjust_gewald_acc();
    double newton_raphson_f_acc();
    double derivf_acc();
    double final_accuracy_acc();

    void compute_gf_denom_acc();
    virtual void compute_gf_ik_acc();
    virtual void compute_gf_ad_acc();
    void compute_sf_precoeff_acc();
    double gf_denom_acc(double, double, double);
    double diffpr_acc(double, double, double, double, double **);

    void compute_rho_coeff_acc();
    void clear_density_acc();
    void load_env_density_acc();
    void make_rho_acc();

    void poisson_energy_acc(int);

    void brick2fft_acc();

    void map2density_one_acc(int);
    void map2density_one_subtract_acc(int);
    void compute_rho1d_acc(const FFT_SCALAR &, const FFT_SCALAR &,
		     const FFT_SCALAR &);

    void compute(int, int);
    void compute_env(int);
    void compute_cplx(int);
    void compute_exch(int);
    void compute_eff(int);

  public:
    int order_acc;
    double g_ewald_acc;
    double shift_acc,shiftone_acc;

    int nx_pppm_acc, ny_pppm_acc, nz_pppm_acc;
    int nxlo_in_acc, nxhi_in_acc, nylo_in_acc, nyhi_in_acc, nzlo_in_acc, nzhi_in_acc;
    int nxlo_out_acc, nxhi_out_acc, nylo_out_acc, nyhi_out_acc, nzlo_out_acc, nzhi_out_acc;
    int nxlo_ghost_acc, nxhi_ghost_acc, nylo_ghost_acc, nyhi_ghost_acc, nzlo_ghost_acc, nzhi_ghost_acc;
    int nxlo_fft_acc, nxhi_fft_acc, nylo_fft_acc, nyhi_fft_acc, nzlo_fft_acc, nzhi_fft_acc;
    int nlower_acc,nupper_acc;
    int ngrid_acc,nfft_acc,nbuf_acc,nfft_both_acc;
    double delxinv_acc,delyinv_acc,delzinv_acc,delvolinv_acc;

    FFT_SCALAR ***env_density_brick_acc;
    FFT_SCALAR ***density_brick_acc;
    FFT_SCALAR ***vdx_brick_acc, ***vdy_brick_acc, ***vdz_brick_acc;
    FFT_SCALAR ***u_brick_acc;

    FFT_SCALAR *density_fft_acc;
    double *greensfn_acc;
    FFT_SCALAR *work1_acc, *work2_acc;
    double **vg_acc;
    double *fkx_acc, *fky_acc, *fkz_acc;

    double *gf_b_acc;
    FFT_SCALAR **rho1d_acc, **rho_coeff_acc, **drho1d_acc, **drho_coeff_acc;
    double *sf_precoeff1_acc, *sf_precoeff2_acc, *sf_precoeff3_acc;
    double *sf_precoeff4_acc, *sf_precoeff5_acc, *sf_precoeff6_acc;
    double sf_coeff_acc[6];          // coefficients for calculating ad self-forces

    class FFT3d *fft1_acc, *fft2_acc;
    class Remap *remap_acc;
    class GridComm *cg_acc;

    FFT_SCALAR * cg_buf1_acc;
    FFT_SCALAR * cg_buf2_acc;

    double **part2grid_dr_acc;
    int **part2grid_acc;

    // Communication functions (based on new GridComm logic)
    MPI_Request request;
    MPI_Status status;

    void reverse_comm_acc(int);
  
    virtual void pack_reverse_acc(int, FFT_SCALAR *, int, int *);
    virtual void unpack_reverse_acc(int, FFT_SCALAR *, int, int *);
};

}

#endif
#endif
