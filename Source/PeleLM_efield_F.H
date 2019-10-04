// Fill bndy funct for ne
  void ne_fill (amrex::Real* data, ARLIM_P(data_lo), ARLIM_P(data_hi),
                const int* dom_lo, const int* dom_hi,
                const amrex::Real* dx, const amrex::Real* grd_lo,
                const amrex::Real* time, const int* bc);

// Fill bndy funct for phiV
  void phiv_fill (amrex::Real* data, ARLIM_P(data_lo), ARLIM_P(data_hi),
                  const int* dom_lo, const int* dom_hi,
                  const amrex::Real* dx, const amrex::Real* grd_lo,
                  const amrex::Real* time, const int* bc);

// Compute RHS of phiV Poisson eq: charge distribution over permittivity 
  void ef_calc_rhs_poisson(const int* lo,  const int* hi,
                           amrex::Real* rhs,          ARLIM_P(rhslo),   ARLIM_P(rhshi),
                           const amrex::Real* rhoY,   ARLIM_P(rYlo),    ARLIM_P(rYhi),
                           const amrex::Real* ne,     ARLIM_P(nelo),    ARLIM_P(nehi));

// Get species mobility Kappa_sp from temperature and species diffusivity using Einstein relation
  void ef_spec_mobility(const int* lo, const int* hi,
                        const amrex::Real* tempdat, ARLIM_P(tlo),  ARLIM_P(thi),
                        const amrex::Real* rhoYdat, ARLIM_P(rylo), ARLIM_P(ryhi),
                        const amrex::Real* Diffdat, ARLIM_P(slo),  ARLIM_P(shi),
                        amrex::Real* kpspdat      , ARLIM_P(kplo), ARLIM_P(kphi));

// Get electron mobility Kappa_e : cst value for now
  void ef_elec_mobility(const int* lo, const int* hi,
                        const amrex::Real* tempdat, ARLIM_P(tlo),   ARLIM_P(thi),
                        const amrex::Real* rhoYdat, ARLIM_P(rylo),  ARLIM_P(ryhi),
                        const amrex::Real* phiVdat, ARLIM_P(plo),   ARLIM_P(phi),
                        amrex::Real* kpedat,        ARLIM_P(kpelo), ARLIM_P(kpehi));

// Get electron diffusivity D_e fron Einstein relation for now
  void ef_elec_diffusivity(const int* lo, const int* hi,
                           const amrex::Real* tempdat, ARLIM_P(tlo),   ARLIM_P(thi),
                           const amrex::Real* rhoYdat, ARLIM_P(rylo),  ARLIM_P(ryhi),
                           const amrex::Real* phiVdat, ARLIM_P(plo),   ARLIM_P(phi),
                           const amrex::Real* kpedat,  ARLIM_P(kpelo), ARLIM_P(kpehi),
                           amrex::Real* dedat,         ARLIM_P(delo),  ARLIM_P(dehi));

// Compute background charge distribution for PNP solve: provisional ions fields / q_e
  void ef_calc_chargedist_prov(const int* lo,  const int* hi,
                               const amrex::Real* rhoYold,  ARLIM_P(rYlo),  ARLIM_P(rYhi),
                               const amrex::Real* Aspec,    ARLIM_P(alo),   ARLIM_P(ahi),
                               const amrex::Real* Dnspec,   ARLIM_P(Dnlo),  ARLIM_P(Dnhi),
                               const amrex::Real* Dnp1spec, ARLIM_P(Dnplo), ARLIM_P(Dnphi),
                               const amrex::Real* Dhatspec, ARLIM_P(Dhlo),  ARLIM_P(Dhhi),
                               const amrex::Real* IRspec,   ARLIM_P(ilo),   ARLIM_P(ihi),
                               amrex::Real* bg_chrg,  ARLIM_P(bglo),  ARLIM_P(bghi),
                               const amrex::Real* dt);

  void ef_pphys_RRATERHOY(const int* lo, const int* hi,
                          const amrex::Real* RhoY, ARLIM_P(RhoYlo),    ARLIM_P(RhoYhi),
                          const amrex::Real* nE_dat, ARLIM_P(nElo),    ARLIM_P(nEhi),
                          const amrex::Real* RhoH, ARLIM_P(RhoHlo),    ARLIM_P(RhoHhi),
                          const amrex::Real* T,    ARLIM_P(Tlo),       ARLIM_P(Thi),
                          amrex::Real*    RhoYdot, ARLIM_P(RhoYdotlo), ARLIM_P(RhoYdothi) );

  int pphys_get_spec_chrg(int* ispec);

  void floor_nE(const int* lo,  const int* hi,
                const amrex::Real* nE_ar,   ARLIM_P(nElo),    ARLIM_P(nEhi));

// Compute lorentz forces for each space dir.
  void ef_lorentz(const int* lo,  const int* hi,
                  const amrex::Real* rhoY,     ARLIM_P(rhoYlo),     ARLIM_P(rhoYhi),
                  const amrex::Real* ne,       ARLIM_P(nelo),       ARLIM_P(nehi),
                  const amrex::Real* xgradPhi, ARLIM_P(xgradPhilo), ARLIM_P(xgradPhihi),
                  const amrex::Real* ygradPhi, ARLIM_P(ygradPhilo), ARLIM_P(ygradPhihi),
#if AMREX_SPACEDIM == 3
                  const amrex::Real* zgradPhi, ARLIM_P(zgradPhilo), ARLIM_P(zgradPhihi),
#endif
                  amrex::Real* lorentz,        ARLIM_P(lorentzlo),  ARLIM_P(lorentzhi) );

// Compute the min dt
   void ef_minTimeStep(const int* lo,  const int* hi,
                       const amrex::Real* U,    ARLIM_P(Ulo),    ARLIM_P(Uhi),
                       const amrex::Real* xUdr, ARLIM_P(xUdrlo), ARLIM_P(xUdrhi),
                       const amrex::Real* yUdr, ARLIM_P(yUdrlo), ARLIM_P(yUdrhi),
#if AMREX_SPACEDIM == 3
                       const amrex::Real* zUdr, ARLIM_P(zUdrlo), ARLIM_P(zUdrhi),
#endif
                       const amrex::Real* dx, amrex::Real* cfl, amrex::Real* dt);

  void getScalingLap(amrex::Real* scaling);
