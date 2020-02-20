// PeleLM efield header. Is included in PeleLM.H if efield requested.
#ifndef _Efield_H_
#define _Efield_H_

// Additional scalar indices
   static int         nE;
   static int         iE_sp;
   static int         PhiV;
   static int         have_nE;
   static int         have_PhiV;

   static amrex::Vector<int>          spec_charges;
   static amrex::BCRec                phiV_bc;
   static amrex::Vector<std::string>  phiV_lo_pol;
   static amrex::Vector<std::string>  phiV_hi_pol;

// Iterative solver options
   static int                  ef_PoissonMaxIter;
   static int                  ef_PoissonVerbose;
   static int                  ef_PoissonMaxOrder;
   static amrex::Real          ef_phiV_tol;
   static amrex::Real          ef_lambda_jfnk;
   static int                  ef_max_GMRES_rst;
   static amrex::Real          ef_GMRES_reltol;
   static int                  ef_GMRES_size;
   static amrex::Real          ef_PC_MG_tol;
   static int                  ef_PC_MG_fixedIte;
   static int                  ef_max_NK_ite;
   static bool                 ef_use_hypre_PCdiff; 

// PNP Data
   amrex::MultiFab             kappaSpec_cc;
   amrex::MultiFab             kappaElec_cc;
   amrex::MultiFab             diffElec_cc;
   amrex::MultiFab**           kappaElec_ec;
   amrex::MultiFab**           diffElec_ec;

   amrex::MultiFab*            grad_phiV_old;

   amrex::MultiFab             pnp_dU;
   amrex::MultiFab             pnp_bgchrg;
   amrex::MultiFab             pnp_gdnv;
   amrex::MultiFab             ne_old;
   amrex::MultiFab             phiV_old;
   amrex::MultiFab*            pnp_Ueff;
   amrex::MultiFab             pnp_refGC;

   amrex::MultiFab*            Udrift_spec;

// PC data  
   amrex::MLABecCecLaplacian   pnp_pc_diff;
   amrex::MLABecLaplacian      pnp_pc_Stilda;
   amrex::MLABecLaplacian      pnp_pc_drift;

   amrex::Real                 pnp_SFne;
   amrex::Real                 pnp_SFphiV;
   amrex::Real                 pnp_SUne;
   amrex::Real                 pnp_SUphiV;

   int                         MGitcount;

// Functions
   static void ef_init();

// Define some multifabs for transport properties
   void ef_define_data();

// SDC predictor : solve for PhiV with 'old' state and compute ef transport properties
   void ef_advance_setup(const amrex::Real &time);

// Compute EF related transport properties: Kp_sp, Kp_e, Diff_e
   void ef_calc_transport(const amrex::Real &time);

// Poisson solve for PhiV with current charge distribution
   void ef_solve_phiv(const amrex::Real &time);

// Compute species drift velocity
   void ef_calcUDriftSpec(const amrex::Real &time);

// Solve PNP function: return updated PhiV and nE
   void ef_solve_PNP(const        int       &misdc,
                     const amrex::Real      &dt,
                     const amrex::Real      &time, 
                     const amrex::MultiFab  &Dn,
                     const amrex::MultiFab  &Dnp1,
                     const amrex::MultiFab  &Dhat,
                           amrex::MultiFab  &ForcingnE);

// PNP NL residual function
   void ef_NL_residual(const amrex::Real        &dt,
                       const amrex::MultiFab    &pnp_U,
                             amrex::MultiFab    &pnp_res,
                             bool update_scaling = false,
                             bool update_PC = false);

// Test PNP NL residual function
   void ef_NL_residual_test(const amrex::Real       &dt,
                            const amrex::MultiFab   &pnp_U,
                                  amrex::MultiFab   &pnp_res,
                                  bool update_scaling = false);

// GMRES solve function
   void ef_GMRES_solve(const amrex::Real      &dt,
                       const amrex::Real      &norm_U,
                       const amrex::MultiFab  &U,
                       const amrex::MultiFab  &res,
                             amrex::MultiFab  &dU,
                             int              &nGMRES);

// J times v function
   void ef_JtV(const amrex::Real      &dt,
               const amrex::Real      &norm_pnp_U,
               const amrex::MultiFab  &pnp_U,
               const amrex::MultiFab  &pnp_res,
               const amrex::MultiFab  &v_in,
                     amrex::MultiFab  &JtV);

// Preconditioner set-up function. Called once per newton ite. TODO: maybe less ?
   void ef_setUpPrecond(const amrex::Real      &dt,
                        const amrex::MultiFab  &pnp_U,
                              amrex::MultiFab  **De_ec);

// Apply the preconditioner to any vector v.
   void ef_applyPrecond(const int              &GMRES_ite,
                        const amrex::MultiFab  &v,
                              amrex::MultiFab  &Pv);

// Residual norm
   amrex::Real ef_NL_norm(const amrex::MultiFab   &pnp_vec);
   void ef_NL_norm(const amrex::MultiFab             &pnp_vec,
                         amrex::Vector<amrex::Real>  &norm);

// PNP provisional charge distribution
   void ef_bg_chrg(const amrex::Real      &dt,
                   const amrex::MultiFab  &Dn,
                   const amrex::MultiFab  &Dnp1,
                   const amrex::MultiFab  &Dhat);

// PNP exit Newton test
   void test_exit_newton(const amrex::MultiFab &pnp_U,
                        const int             &NK_ite,
                        const amrex::Real     &norm0,
                        const amrex::Real     &norm,
                        bool                  & exit_newton);

// PNP Newton linesearch
   void ef_linesearch(const amrex::Real     &dt,
                           amrex::MultiFab &pnp_U,
                           amrex::MultiFab &pnp_dU,
                           amrex::MultiFab &pnp_res,
                           amrex::Real     &norm_U,
                           amrex::Real     &norm_res);

// Get electron edge transport properties
   void ef_get_edge_transport(amrex::MultiFab** ke_ec,
                             amrex::MultiFab** De_ec);

// Compute the electron diffusion term
   void compute_ne_diffusion_term(const amrex::Real       &dt,
                                  const amrex::MultiFab   &pnp_U,
                                        amrex::MultiFab** De_ec,
                                        amrex::MultiFab   &diff_ne_term);

// Compute the electron convection term
   void compute_ne_convection_term(const amrex::Real       &dt,
                                         amrex::MultiFab   &pnp_U,
                                         amrex::MultiFab** Ke_ec,
                                         amrex::MultiFab** phiV_fluxes,
                                         amrex::MultiFab   &conv_ne_term);

// Compute the phiV Laplacian term and the PhiV 'fluxes'
   void compute_phiV_laplacian_term(const amrex::Real       &dt,
                                    const amrex::MultiFab   &pnp_U,
                                          amrex::MultiFab** phiV_fluxes,
                                          amrex::MultiFab   &phiV_laplacian_term);

// Compute the phiV Laplacian term and the PhiV 'fluxes'
   void ef_calcGradPhiV(amrex::MultiFab   &phiV,
                        amrex::MultiFab   **grad_phiV);

   void ef_reactionRateRhoY_pphys(amrex::FArrayBox&       RhoYdot,
                                  const amrex::FArrayBox& RhoY,
                                  const amrex::FArrayBox& RhoH,
                                  const amrex::FArrayBox& T,
                                  const amrex::FArrayBox& ne_fab,
                                  const amrex::Box&       box,
                                  int              sCompRhoY,
                                  int              sCompRhoH,
                                  int              sCompT,
                                  int              sCompnE,
                                  int              sCompRhoYdot) const;

// Fill the species charges vector
   static void getSpeciesCharges(amrex::Vector<int>& spec_chrg);

   static amrex::BCRec hack_bc_charged_spec(const int &chrg,
                                            const amrex::BCRec &bc_in);

// Compute the Lorentz forces used in the momemtum eq.
   void ef_calcLorentz(const amrex::Real       &time,
                             amrex::MultiFab   &lorentz);

// Time step estimate
   amrex::Real ef_estTimeStep();

// Similar to center_to_edge_fancy but upwinded
   static void center_to_edge_upwind_fancy (const amrex::FArrayBox& cfab,
                                            const amrex::FArrayBox& velefab,
                                            amrex::FArrayBox&       efab,
                                            const amrex::Box&       ccBox,
                                            const amrex::Box&       eBox,
                                            int              sComp,
                                            int              dComp,
                                            int              nComp,
                                            const amrex::Box&       domain,
                                            const FPLoc&     bc_lo,
                                            const FPLoc&     bc_hi);

// Setup ne BC for diffusion operator
   void ef_set_neBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &diff_lobc,
                    std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &diff_hibc);

// Setup PhiV BC for linear Poisson solve
   void ef_set_PoissonBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                         std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc);

// Setup BC for Precond: all Neumann or Perio
   void ef_set_PCBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                    std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc);

#endif /*_Efield_H_*/
