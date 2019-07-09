void PeleLM::ef_advance_setup(const Real &time) {

   ef_solve_phiv(time); 

	ef_calc_transport(time);
}	

void PeleLM::ef_calc_transport(const Real &time) {
	BL_PROFILE("EF::ef_calc_transport()");

	const TimeLevel whichTime = which_time(State_Type, time);

	BL_ASSERT(whichTime == AmrOldTime || whichTime == AmrNewTime);

//	Get ptr to current version of diff	
	MultiFab&  diff      = (whichTime == AmrOldTime) ? (*diffn_cc) : (*diffnp1_cc);
   const int nGrow      = diff.nGrow();
 
//	A few check on the grow size of EF transport properties
   BL_ASSERT(kappaSpec_cc.nGrow() >= nGrow);
   BL_ASSERT(kappaElec_cc.nGrow() >= nGrow);
   BL_ASSERT(diffElec_cc.nGrow()  >= nGrow);

//	FillPatchIterator for State_type variables. Is it necessary ? For specs, it's been done before for 
//	other transport properties
   FillPatchIterator rhoY_fpi(*this,diff,nGrow,time,State_Type,first_spec,nspecies);
   FillPatchIterator T_fpi(*this,diff,nGrow,time,State_Type,Temp,1);
   FillPatchIterator PhiV_fpi(*this,diff,nGrow,time,State_Type,PhiV,1);
   MultiFab& rhoYmf=rhoY_fpi.get_mf();
   MultiFab& Tmf=T_fpi.get_mf();
   MultiFab& PhiVmf=PhiV_fpi.get_mf();

// Call Fortran to effectively compute the transport properties
// Some of the Fortran should probably be in ChemDriver, but keep it here for now ...	
#ifdef _OPENMP
#pragma omp parallel
#endif  
  for (MFIter mfi(rhoYmf,true); mfi.isValid();++mfi)
  {
     const FArrayBox& RhoD = diff[mfi];
     const FArrayBox& RhoYfab = rhoYmf[mfi];
     const FArrayBox& Tfab = Tmf[mfi];
     const FArrayBox& PhiVfab = PhiVmf[mfi];
     FArrayBox& Kpspfab = kappaSpec_cc[mfi];
     FArrayBox& Kpefab = kappaElec_cc[mfi];
     FArrayBox& Diffefab = diffElec_cc[mfi];
     const Box& gbox = mfi.growntilebox();

     ef_spec_mobility(gbox.loVect(),gbox.hiVect(),
                      Tfab.dataPtr(),    ARLIM(Tfab.loVect()),    ARLIM(Tfab.hiVect()),
                      RhoYfab.dataPtr(), ARLIM(RhoYfab.loVect()), ARLIM(RhoYfab.hiVect()),
                      RhoD.dataPtr(),    ARLIM(RhoD.loVect()),    ARLIM(RhoD.hiVect()),
                      Kpspfab.dataPtr(), ARLIM(Kpspfab.loVect()), ARLIM(Kpspfab.hiVect()));

     ef_elec_mobility(gbox.loVect(),gbox.hiVect(),
                      Tfab.dataPtr(),    ARLIM(Tfab.loVect()),    ARLIM(Tfab.hiVect()),
                      RhoYfab.dataPtr(), ARLIM(RhoYfab.loVect()), ARLIM(RhoYfab.hiVect()),
                      PhiVfab.dataPtr(), ARLIM(PhiVfab.loVect()), ARLIM(PhiVfab.hiVect()),
                      Kpefab.dataPtr(),  ARLIM(Kpefab.loVect()),  ARLIM(Kpefab.hiVect()));

     ef_elec_diffusivity(gbox.loVect(),gbox.hiVect(),
                         Tfab.dataPtr(),     ARLIM(Tfab.loVect()),     ARLIM(Tfab.hiVect()),
                         RhoYfab.dataPtr(),  ARLIM(RhoYfab.loVect()),  ARLIM(RhoYfab.hiVect()),
                         PhiVfab.dataPtr(),  ARLIM(PhiVfab.loVect()),  ARLIM(PhiVfab.hiVect()),
                         Kpefab.dataPtr(),   ARLIM(Kpefab.loVect()),   ARLIM(Kpefab.hiVect()),
							    Diffefab.dataPtr(), ARLIM(Diffefab.loVect()), ARLIM(Diffefab.hiVect()));
  }
}

void PeleLM::ef_define_data() {
   kappaSpec_cc.define(grids,dmap,nspecies,1);
   kappaElec_cc.define(grids,dmap,1,1);
   diffElec_cc.define(grids,dmap,1,1);
	kappaElec_ec = 0;
	diffElec_ec  = 0;

	grad_phiV_old = 0;

	pnp_dU.define(grids,dmap,2,1);
	pnp_bgchrg.define(grids,dmap,1,1);
	pnp_gdnv.define(grids,dmap,1,1);
	ne_old.define(grids,dmap,1,1);
	phiV_old.define(grids,dmap,1,1);
	pnp_refGC.define(grids,dmap,2,Godunov::hypgrow());

	pnp_Ueff = new MultiFab[AMREX_SPACEDIM];
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		const BoxArray& edgeba = getEdgeBoxArray(d);
	   pnp_Ueff[d].define(edgeba,dmap, 1,1);
	}

	pnp_SFne = 1.0;
	pnp_SFphiV = 1.0;
	pnp_SUne = 1.0;
	pnp_SUphiV = 1.0;
}

void PeleLM::ef_solve_phiv(const Real &time) {

	MultiFab& S = get_new_data(State_Type);
	MultiFab rhs_poisson(grids,dmap,1,nGrowAdvForcing);

// Use FillPatchIterator (FPI) to update the data in the growth cell and copy back into S
	FillPatchIterator PhiVfpi(*this,S,1,time,State_Type,PhiV,1);
	MultiFab& PhiVmf = PhiVfpi.get_mf();
	MultiFab::Copy(S,PhiVmf,0,PhiV,1,1);
							
// Get alias to PhiV in S
	MultiFab PhiV_alias(S,amrex::make_alias,PhiV,1);

// Get RHS		
#ifdef _OPENMP
#pragma omp parallel
#endif
   for (MFIter mfi(S,true); mfi.isValid(); ++mfi)
   {
      const Box& box = mfi.tilebox();
      const FArrayBox& rhoY = S[mfi];
      const FArrayBox& ne = S[mfi];
      FArrayBox& rhs = rhs_poisson[mfi];
      ef_calc_rhs_poisson(box.loVect(), box.hiVect(),
                         rhs.dataPtr(0),           ARLIM(rhs.loVect()),    ARLIM(rhs.hiVect()),
					          rhoY.dataPtr(first_spec), ARLIM(rhoY.loVect()),   ARLIM(rhoY.hiVect()),
					          ne.dataPtr(nE),           ARLIM(ne.loVect()),     ARLIM(ne.hiVect()));
   }

// Set-up solver tolerances
   const Real S_tol     = ef_phiV_tol;
   const Real S_tol_abs = rhs_poisson.norm0() * ef_phiV_tol;

// Set-up Poisson solver
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);

   MLPoisson phiV_poisson({geom}, {grids}, {dmap}, info);

   phiV_poisson.setMaxOrder(ef_PoissonMaxOrder);

// Set-up BC's
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
	ef_set_PoissonBC(mlmg_lobc, mlmg_hibc);
   phiV_poisson.setDomainBC(mlmg_lobc, mlmg_hibc);
   phiV_poisson.setLevelBC(0, &PhiV_alias);

// LinearSolver options
	MLMG mlmg(phiV_poisson);
	mlmg.setMaxIter(ef_PoissonMaxIter);
	mlmg.setMaxFmgIter(30);
	mlmg.setVerbose(ef_PoissonVerbose);
//	mlmg.setBottomVerbose(bottom_verbose);

// Actual solve
	mlmg.solve({&PhiV_alias}, {&rhs_poisson}, S_tol, S_tol_abs);

}

void PeleLM::ef_init() {
	amrex::Print() << " Init EFIELD solve options \n" << "\n";
	
	PeleLM::nE                        = -1;
	PeleLM::PhiV                      = -1;
	PeleLM::have_nE                   = 0;
	PeleLM::have_PhiV                 = 0;
	PeleLM::ef_phiV_tol               = 1.0e-7;
	PeleLM::ef_PC_MG_tol              = 1.0e-10;
	PeleLM::ef_PoissonMaxIter			 = 1000;
	PeleLM::ef_PoissonVerbose			 = 1;
	PeleLM::ef_PoissonMaxOrder			 = 4;
	PeleLM::ef_max_NK_ite             = 20;
	PeleLM::ef_lambda_jfnk				 = 1.0e-7;
	PeleLM::ef_max_GMRES_rst			 = 50;
	PeleLM::ef_GMRES_reltol  			 = 1.0e-10;
	PeleLM::ef_GMRES_size			    = 10;
	
	ParmParse pp("ef");
	
	pp.query("MG_itermax",ef_PoissonMaxIter);
	pp.query("MG_verbose",ef_PoissonVerbose);
	pp.query("MG_maxorder",ef_PoissonMaxOrder);
	pp.query("MG_PhiV_tol",ef_phiV_tol);
	pp.query("PC_MG_tol",ef_PC_MG_tol);
	pp.query("JFNK_lambda",ef_lambda_jfnk);
	pp.query("JFNK_maxNewton",ef_max_NK_ite);
	pp.query("GMRES_max_restart",ef_max_GMRES_rst);
	pp.query("GMRES_rel_tol",ef_GMRES_reltol);
	pp.query("GMRES_restart_size",ef_GMRES_size);
}


void PeleLM::ef_solve_PNP(const Real     &dt, 
							     const Real     &time, 
							     const MultiFab &Dn,
							     const MultiFab &Dnp1,
							     const MultiFab &Dhat) {

	BL_PROFILE("EF::ef_solve_PNP()");
	
	//	TODO: Compute the old diffusion term for the Godunov Forcing
	pnp_gdnv.setVal(0.0);	
	
	// Get edge-averaged transport properties. Edge data is "shared"
	FluxBoxes diff_e(this, 1, 1);
	FluxBoxes conv_e(this, 1, 1);
	kappaElec_ec = conv_e.get();
	diffElec_ec = diff_e.get();
	ef_get_edge_transport(kappaElec_ec, diffElec_ec); 
	
	// Define and get PNP components
	MultiFab&  S = get_new_data(State_Type);
	MultiFab&  S_old = get_old_data(State_Type);
	MultiFab::Copy(ne_old,S_old,nE,0,1,1);
	MultiFab::Copy(phiV_old,S_old,PhiV,0,1,1);
	
	//	Define a FPI to get the (max) GC used throughout the PNP resolution
	//	I'll need at most 4 GC for the Godunov stuff
	FillPatchIterator nEfpi(*this,S,Godunov::hypgrow(),time,State_Type,nE,2);
	MultiFab& nEfpi_mf = nEfpi.get_mf();
	MultiFab::Copy(pnp_refGC, nEfpi_mf, 0, 0, 2, Godunov::hypgrow());
	
	// GC of pnp_U init with the GC from S. Should'nt change during PNP solve ...	
	MultiFab pnp_U(grids,dmap,2,Godunov::hypgrow());
	MultiFab pnp_res(grids,dmap,2,Godunov::hypgrow());
	MultiFab::Copy(pnp_U, pnp_refGC, 0, 0, 2, Godunov::hypgrow());
	
	// Get the NL vector scaling and scale
	pnp_SUne = pnp_U.norm0(0);
	pnp_SUphiV = pnp_U.norm0(1);
	pnp_U.mult(1.0/pnp_SUne,0,1);	
   pnp_U.mult(1.0/pnp_SUphiV,1,1);	
	amrex::Print() << " ne scaling: " << pnp_SUne << "\n";
	amrex::Print() << " PhiV scaling: " << pnp_SUphiV << "\n";
	showMF("pnp",pnp_U,"pnp_U0",level);
	showMFsub("pnp1D",pnp_U,stripBox,"1Dpnp_U0",level);

	//	Vector<Real> norm_NL_U0(2);
	const Real norm_NL_U0 = ef_NL_norm(pnp_U); 

	// Compute provisional CD
   ef_bg_chrg(dt, Dn, Dnp1, Dhat);

	// Need the gradient of phiV_old
   FluxBoxes fluxb(this, 1, 1);
   grad_phiV_old = fluxb.get();
	ef_calc_grad(dt, phiV_old, grad_phiV_old);

	// Pre-Newton stuff	
	// first true trigger initalize the residual scaling.
	// second true trigger initalize the preconditioner.
   ef_NL_residual( dt, pnp_U, pnp_res, true, true );
	showMF("pnp",pnp_res,"pnp_res0",level);
	showMF("pnp",pnp_bgchrg,"pnp_bgchrg",level);
//	showMFsub("pnp1D",pnp_res,stripBox,"1Dpnp_res0",level);
	pnp_res.mult(-1.0,0,2);
   const Real norm_NL_res0 = ef_NL_norm(pnp_res);	
	// TODO: Check for direct convergence
	// TODO: Get data for globalization algo: in 1D I call Jac ... not great ...

	Real norm_NL_res = norm_NL_res0;
	Real norm_NL_U = norm_NL_U0;

   bool exit_newton = false;
   int NK_ite = 0;	
   do {

	   NK_ite += 1;
	   amrex::Print() << " Newton it: " << NK_ite << " residual: " << 0.5*norm_NL_res*norm_NL_res << "\n";

		//    GMRES 
      ef_GMRES_solve( dt, norm_NL_U, pnp_U, pnp_res, pnp_dU ); 	

		//    Linesearch
		ef_linesearch( dt, pnp_U, pnp_dU, pnp_res, norm_NL_U, norm_NL_res);

		//    Test exit conditions  	
	   test_exit_newton(pnp_res, NK_ite, norm_NL_res0, norm_NL_res, exit_newton);

//		amrex::Abort("In Newton stop");
   } while( ! exit_newton );

	// Post newton stuff
	showMFsub("pnp1D",pnp_U,stripBox,"1DpostNewt_U",level);
	showMF("pnp",pnp_U,"postNewt_U",level);
	//TODO: 0 or 1 GC after pnp ?
	MultiFab::Copy(S,pnp_U,0,nE,2,1);

	amrex::Abort("Post Newton stop");
}

Real PeleLM::ef_NL_norm(const MultiFab &pnp_vec) {
	Real norm = MultiFab::Dot(pnp_vec,0,pnp_vec,0,1,0) +
               MultiFab::Dot(pnp_vec,1,pnp_vec,1,1,0);
	norm = std::sqrt(norm);	 		
	return norm;
}

void PeleLM::ef_linesearch(const Real		&dt,
									MultiFab 		&pnp_U,
									MultiFab 		&pnp_dU,
									MultiFab 		&pnp_res,
									Real		 		&norm_U,
								   Real		 		&norm_res) {

		// Backtracking params
		Real alpha = 0.0001;
		Real lambdared_min = 0.1;
		Real lambdared_max = 0.5;	
		int  max_ls_its = 10;

		MultiFab pnp_Utmp(grids,dmap,2,Godunov::hypgrow());
		MultiFab pnp_restmp(grids,dmap,2,0);
		pnp_restmp.setVal(0.0);
		pnp_Utmp.setVal(0.0);

		// Backtracking algo from Dennis & Schnabel
		// Objective function : 0.5 * norm(F)**2
		Real obj_old = 0.5*norm_res*norm_res;
		Real norm_LS_res, obj_new, sufficient_reduction;	
		Real lambdatmp, lambdaprv, obj_newprv;
		Real r, rprv, a, b, disc;

		// Need the initslope
      ef_JtV(dt, norm_U, pnp_U, pnp_res, pnp_dU, pnp_Utmp);		//JtV on dU -> Utmp	
		Real initslope = MultiFab::Dot(pnp_Utmp,0,pnp_res,0,2,0);// 
		if (initslope > 0.0)  initslope = -initslope;
		if (initslope == 0.0) initslope = -1.0;

	   //amrex::Print() << " Initslope: " << initslope << "\n";
		
		bool done_linesearch = false;
      Real lambda = 1.0; 
		int  ls_its = 0;
   	do {
			//MultiFab::LinComb(pnp_Utmp,1.0,pnp_U,0,lambda,pnp_dU,0,0,2,Godunov::hypgrow());
		   //showMFsub("pnp1D",pnp_U,stripBox,"1Dpnp_ULS",level);
		   //showMFsub("pnp1D",pnp_dU,stripBox,"1Dpnp_dULS",level);
		   //showMFsub("pnp1D",pnp_Utmp,stripBox,"1Dpnp_UtmpLS",level);
			MultiFab::Copy(pnp_Utmp,pnp_dU,0,0,2,0);
			pnp_Utmp.mult(lambda,0,2);
			pnp_Utmp.plus(pnp_U,0,2,Godunov::hypgrow());
			ef_NL_residual( dt, pnp_Utmp, pnp_res, false, true );
	   	pnp_res.mult(-1.0,0,2);
		   //showMFsub("pnp1D",pnp_res,stripBox,"1Dpnp_resLS",level);
		   //showMFsub("pnp1D",pnp_restmp,stripBox,"1Dpnp_restmpLS",level);
			norm_LS_res = ef_NL_norm(pnp_res);
			obj_new = 0.5*norm_LS_res*norm_LS_res;
			sufficient_reduction = obj_old + lambda*alpha*initslope;
		   //amrex::Abort("Abort in linesearch");
			//if ( obj_new <= sufficient_reduction ) {
			//	done_linesearch = true;
			//	if ( lambda == 1.0 ) amrex::Print() << " Taking full Newton step \n";
			//}	else {
			//	amrex::Print() << " lambda: " << lambda << " . obj_new: " << obj_new << " .Red: " << sufficient_reduction << "\n";  
			//	if ( lambda == 1.0 ) {
			//		lambdatmp = -initslope / ( 2.0 * ( obj_new - obj_old - initslope) );
			//	} else {
			//		r = (obj_new - obj_old - lambda*initslope)/(lambda*lambda);
			//		rprv = (obj_newprv - obj_old - lambdaprv*initslope)/(lambdaprv*lambdaprv);
			//		a = (r - rprv)/(lambda - lambdaprv);
			//		b = (-lambdaprv*r + lambda*rprv)/(lambda - lambdaprv);	
			//		if ( a == 0.0 ) {
			//			lambdatmp = -initslope / ( 2.0 * b );
			//		} else {
			//			disc = b*b - 3.0*a*initslope;
			//			if (disc < 0.0) disc = 0.0;
			//			lambdatmp = ( - b + pow(disc,0.5) ) / (3.0*a);
			//		}
			//	}
			//	lambdaprv = lambda;
			//	obj_newprv = obj_new;
			//	if (lambdatmp >  lambdared_max*lambda) lambdatmp = lambdared_max*lambda;
			//  if (lambdatmp <= lambdared_min*lambda) lambdatmp = lambdared_min*lambda;	
			//	lambda = lambdatmp;
			//}
			//ls_its += 1;
			//if ( ls_its >= max_ls_its ) done_linesearch = true;
			done_linesearch = true;
		} while (!done_linesearch);

		//amrex::Abort("Abort in linesearch");

		// Update final solution/residual
		MultiFab::Copy(pnp_U,pnp_Utmp, 0, 0, 2, Godunov::hypgrow()); 
		//MultiFab::Copy(pnp_res,pnp_restmp, 0, 0, 2, 0); 
		norm_res = ef_NL_norm(pnp_res);;
		norm_U = ef_NL_norm(pnp_U);

}	


void PeleLM::test_exit_newton(const MultiFab &pnp_res,
										const int      &NK_ite, 
										const Real     &norm_res0,
										const Real     &norm_res,
										      bool     &exit_newton) {

  const Real tol_Newton = pow(1.0e-16,2.0/3.0);
  Real max_res = pnp_res.norm0();
  if ( max_res <= tol_Newton ) {
	  exit_newton = true;
	  amrex::Print() << " Final Newton res: " << max_res << "\n";
	  amrex::Print() << " Converged Newton ite of PNP solve \n";
  }

  if ( NK_ite > ef_max_NK_ite ) {
	  exit_newton = true;
	  amrex::Print() << " Max Newton iteration reached for PNP solve \n";
  }

}

void PeleLM::ef_NL_residual(const Real      &dt,
								    const MultiFab  &pnp_U_in,
									       MultiFab  &pnp_res,
									       bool      update_scaling,
											 bool		  update_PC	) {

   MultiFab& I_R = get_new_data(RhoYdot_Type);
	MultiFab I_R_e(I_R,amrex::make_alias,20,1); // TODO : define iE_sp in C++

// Copy const pnp_U_in into local to unscale pnp_U  
	MultiFab pnp_U(grids,dmap,2,Godunov::hypgrow());
	MultiFab::Copy(pnp_U, pnp_U_in, 0, 0, 2, Godunov::hypgrow());

// Unscale pnp_U. 
   pnp_U.mult(pnp_SUne,0,1);
   pnp_U.mult(pnp_SUphiV,1,1);
 
// Laplacian and fluxes of PhiV
   FluxBoxes fluxb(this, 1, 1);
   MultiFab** phiV_flux = fluxb.get();
	MultiFab laplacian_term(grids, dmap, 1, 0);
   compute_phiV_laplacian_term(dt, pnp_U, phiV_flux, laplacian_term);
//	showMFsub("pnp1D",laplacian_term,stripBox,"1DLapl_phi",level);
	Real scaling_Laplacian = 0.0;
	getScalingLap(&scaling_Laplacian);

// Use amrex operators to get the RHS
// Diffusion of ne
	MultiFab diff_ne_term(grids, dmap, 1, 0);
   compute_ne_diffusion_term(dt, pnp_U, diffElec_ec, diff_ne_term);
//	showMFsub("pnp1D",diff_ne_term,stripBox,"1Ddiff_ne",level);

// Convection of ne
	MultiFab conv_ne_term(grids, dmap, 1, 1);
   compute_ne_convection_term(dt, pnp_U, kappaElec_ec, phiV_flux, conv_ne_term);
//	showMFsub("pnp1D",conv_ne_term,stripBox,"1Dconv_ne",level);
 
// Build the non-linear residual	
// res(ne(:)) = dt * ( diff(:) + conv(:) + I_R(:) ) - ( ne(:) - ne_old(:) )
// res(phiv(:)) = \Sum z_k * \tilde Y_k / q_e - ne + Lapl_PhiV
   for (MFIter mfi(pnp_res,true); mfi.isValid(); ++mfi)
   {
      const Box& box = mfi.tilebox();
      const FArrayBox& Ufab = pnp_U[mfi];
      const FArrayBox& difffab = diff_ne_term[mfi];
      const FArrayBox& convfab = conv_ne_term[mfi];
      const FArrayBox& laplfab = laplacian_term[mfi];
      const FArrayBox& IRefab = I_R_e[mfi];
      const FArrayBox& neoldfab = ne_old[mfi];
      const FArrayBox& bgchargfab = pnp_bgchrg[mfi];
      FArrayBox& Resfab = pnp_res[mfi];

	   Resfab.copy(difffab,box,0,box,0,1);		// Copy diff term into res for ne
	   Resfab.plus(convfab,box,box,0,0,1);		// Add diff term into res for ne
		Resfab.plus(IRefab,box,box,0,0,1);	   // Add forcing term (I_R) into res for ne
		Resfab.mult(dt,box,0,1);					// times dt
		Resfab.minus(Ufab,box,box,0,0,1);		// Substract current ne
		Resfab.plus(neoldfab,box,box,0,0,1);	// Add old ne --> Done with ne residuals

	   Resfab.copy(laplfab,box,0,box,1,1);		// Copy the phiV laplacian into res for phiV
		Resfab.mult(scaling_Laplacian,box,1,1);// times e_0*e_r/q_E
		Resfab.minus(Ufab,box,box,0,1,1);	   // Substract current ne
		Resfab.plus(bgchargfab,box,box,0,1,1);	// Add bg charge term / q_E --> Done with phiV residuals
   }

// Residual scaling	
   if ( update_scaling ) {
		pnp_SFne = pnp_res.norm0(0);
		pnp_SFphiV = pnp_res.norm0(1);
	   amrex::Print() << " F(ne) scaling: " << pnp_SFne << "\n";
	   amrex::Print() << " F(PhiV) scaling: " << pnp_SFphiV << "\n";
	}

	pnp_res.mult(1.0/pnp_SFne,0,1);
	pnp_res.mult(1.0/pnp_SFphiV,1,1);

//	showMF("pnp",pnp_res,"pnp_res",level);

// Update the preconditioner
   if ( update_PC ) {
	   ef_setUpPrecond(dt, pnp_U, diffElec_ec);
	}

}

void PeleLM::ef_NL_residual_test(const Real      &dt,
								         const MultiFab  &pnp_U_in,
									            MultiFab  &pnp_res,
									      bool update_scaling) {

	MultiFab pnp_U(grids,dmap,2,0);
	MultiFab::Copy(pnp_U, pnp_U_in, 0, 0, 2, 0);

// Unscale pnp_U. 
   pnp_U.mult(pnp_SUne,0,1);
   pnp_U.mult(pnp_SUphiV,1,1);

// Dummy residual function : /10	
   MultiFab::Copy(pnp_res, pnp_U, 0, 0, 2, 0);	
	pnp_res.mult(1.0/10.0);

// Residual scaling	
   if ( update_scaling ) {
		pnp_SFne = pnp_res.norm0(0);
		pnp_SFphiV = pnp_res.norm0(1);
	   amrex::Print() << " F(ne) scaling: " << pnp_SFne << "\n";
	   amrex::Print() << " F(PhiV) scaling: " << pnp_SFphiV << "\n";
	}

	pnp_res.mult(1.0/pnp_SFne,0,1);
	pnp_res.mult(1.0/pnp_SFphiV,1,1);

}

void PeleLM::ef_bg_chrg(const Real      &dt,
							   const MultiFab  &Dn,
								const MultiFab  &Dnp1,
								const MultiFab  &Dhat) {

#ifdef _OPENMP
#pragma omp parallel
#endif
   for (MFIter mfi(pnp_bgchrg,true); mfi.isValid(); ++mfi) {
		const Box& box = mfi.tilebox();
	   const FArrayBox& rhoYoldfab = get_old_data(State_Type)[mfi];
	   const FArrayBox& afab = (*aofs)[mfi];
		const FArrayBox& dnfab = Dn[mfi];   
		const FArrayBox& dnp1fab = Dnp1[mfi];
		const FArrayBox& dhatfab = Dhat[mfi];
	   const FArrayBox& rfab = get_new_data(RhoYdot_Type)[mfi];
	   FArrayBox& bgchrgfab = pnp_bgchrg[mfi];
		ef_calc_chargedist_prov(box.loVect(), box.hiVect(),
				                  rhoYoldfab.dataPtr(first_spec), ARLIM(rhoYoldfab.loVect()), ARLIM(rhoYoldfab.hiVect()), 
										afab.dataPtr(first_spec), ARLIM(afab.loVect()),       ARLIM(afab.hiVect()),
										dnfab.dataPtr(0),         ARLIM(dnfab.loVect()),      ARLIM(dnfab.hiVect()),
										dnp1fab.dataPtr(0),       ARLIM(dnp1fab.loVect()),    ARLIM(dnp1fab.hiVect()),
										dhatfab.dataPtr(0),       ARLIM(dhatfab.loVect()),    ARLIM(dhatfab.hiVect()),
										rfab.dataPtr(0),          ARLIM(rfab.loVect()),       ARLIM(rfab.hiVect()),
										bgchrgfab.dataPtr(),		  ARLIM(bgchrgfab.loVect()),  ARLIM(bgchrgfab.hiVect()),
										&dt);
   }
}

void PeleLM::ef_get_edge_transport(MultiFab *Ke_ec[BL_SPACEDIM],
											  MultiFab *De_ec[BL_SPACEDIM]) {

	amrex::Print() << " Computing edge based elec transport prop. \n";
   for (MFIter mfi(diffElec_cc,true); mfi.isValid(); ++mfi) {
	   const Box& box = mfi.validbox();

      for (int dir = 0; dir < BL_SPACEDIM; dir++) {
         FPLoc bc_lo = fpi_phys_loc(get_desc_lst()[State_Type].getBC(nE).lo(dir));
	 	   FPLoc bc_hi = fpi_phys_loc(get_desc_lst()[State_Type].getBC(nE).hi(dir));
	 	   const Box& ebox = mfi.nodaltilebox(dir);
	 	   center_to_edge_fancy((diffElec_cc)[mfi],(*De_ec[dir])[mfi],
		                         amrex::grow(box,amrex::BASISV(dir)), ebox, 0, 
		                         0, 1, geom.Domain(), bc_lo, bc_hi);
	 	   center_to_edge_fancy((kappaElec_cc)[mfi],(*Ke_ec[dir])[mfi],
		                         amrex::grow(box,amrex::BASISV(dir)), ebox, 0, 
		                         0, 1, geom.Domain(), bc_lo, bc_hi);
      }
   }
//   showMF("pnp",*De_ec[0],"pnp_De_ec_x",level);
//   showMF("pnp",*De_ec[1],"pnp_De_ec_y",level);

}

void PeleLM::compute_ne_diffusion_term(const Real      &dt,
													const MultiFab  &pnp_U, 
													      MultiFab  *De_ec[BL_SPACEDIM],
													      MultiFab  &diff_ne) {

// Get nEmf from fpi on S for BC
//	const Real time  = state[State_Type].curTime();	// current time
//	MultiFab&  S = get_new_data(State_Type);
//	FillPatchIterator nEfpi(*this,S,1,time,State_Type,nE,1);
//	MultiFab& nEmf = nEfpi.get_mf();
//	MultiFab nE_new(grids,dmap,1,1);
//	MultiFab::Copy(nE_new,pnp_U,0,0,1,0);
//	nE_new.FillBoundary();
	MultiFab nE_alias(pnp_U,amrex::make_alias,0,1);

// Set-up Lapl operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
	info.setMaxCoarseningLevel(0);
   MLABecLaplacian ne_LAPL({geom}, {grids}, {dmap}, info);
   ne_LAPL.setMaxOrder(ef_PoissonMaxOrder);
	  
// BC's	
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
	ef_set_neBC(mlmg_lobc, mlmg_hibc);
	ne_LAPL.setDomainBC(mlmg_lobc, mlmg_hibc);
   ne_LAPL.setLevelBC(0, &nE_alias);

// Coeff's	
   ne_LAPL.setScalars(0.0, 1.0); 
	std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{D_DECL(De_ec[0],De_ec[1],De_ec[2])};
	ne_LAPL.setBCoeffs(0, bcoeffs);

// Get divergence using apply function	
//	FluxBoxes fluxb  (this, 1, 0);								// Flux box ...
//	MultiFab **flux    =   fluxb.get();							// ... associated flux MultiFab
	MLMG mlmg(ne_LAPL);
	mlmg.apply({&diff_ne},{&nE_alias});

	diff_ne.mult(-1.0);

//	std::array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(flux[0],flux[1],flux[2])};
//	mlmg.getFluxes({fp},{&nEU_alias});

// Rescale fluxes to get right stuff regardless of cartesian or r-Z
//   for (int d = 0; d < BL_SPACEDIM; ++d) 
//      flux[d]->mult(1.0/(geom.CellSize()[d]));   

// Get flux divergence, scaled by -1
//	flux_divergence(diff_ne,0,flux,0,1,-1);

}

void PeleLM::compute_ne_convection_term(const Real      &dt,
													  		 MultiFab  &pnp_U, 
													       MultiFab  *Ke_ec[AMREX_SPACEDIM],
													       MultiFab  *grad_phiV[AMREX_SPACEDIM],
													       MultiFab  &conv_ne) {

	const Real* dx        = geom.CellSize();

// Build u_eff : u_mac - Ke*grad(\phi) at faces	
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		pnp_Ueff[d].setVal(0.0);
		// Build time centered grad PhiV
		grad_phiV[d]->plus(*grad_phiV_old[d],0,1,1);
		grad_phiV[d]->mult(0.5);
   	MultiFab::AddProduct(pnp_Ueff[d],*Ke_ec[d],0,*grad_phiV[d],0,0,1,0);
		pnp_Ueff[d].mult(-geom.CellSize()[d]);
		pnp_Ueff[d].plus(u_mac[d],0,1,u_mac[d].nGrow());
	}
	//showMFsub("pnp1D",pnp_Ueff[1],stripBox,"1DUeff_y",level);

	const Real time  = state[State_Type].curTime();	// current time
	MultiFab nE_new(grids,dmap,1,Godunov::hypgrow());
	MultiFab::Copy(nE_new,pnp_U,0,0,1,0);
	nE_new.FillBoundary();
	// TODO: FillBoundary() only works 'cause its single level right now.
	BL_ASSERT(level==0);

   {
   	FArrayBox cflux[AMREX_SPACEDIM];
   	FArrayBox edgstate[AMREX_SPACEDIM];
   	Vector<int> state_bc;
   	MultiFab DivU(grids,dmap,1,1);
   	DivU.setVal(0.0);

		// Not using the classical aofs and so on, so I need to trick godunov 
		// into using Conservative ne adv	
		Vector<AdvectionForm> advectionTypeLcl;
		advectionTypeLcl.resize(1);
		advectionTypeLcl[0] = Conservative;
   
   	for (MFIter mfi(nE_new,false); mfi.isValid(); ++mfi) {
   		const Box& bx = mfi.validbox();
   		const FArrayBox& neFab = nE_new[mfi];
   		const FArrayBox& force = pnp_gdnv[mfi];
   		const FArrayBox& divu  = DivU[mfi];
   
   		for (int d=0; d<AMREX_SPACEDIM; ++d) {
   			const Box& ebx = amrex::surroundingNodes(bx,d);
   			cflux[d].resize(ebx,1);
   			edgstate[d].resize(ebx,1);
   		}
   		conv_ne[mfi].setVal(0.0);
   
   		state_bc = fetchBCArray(State_Type,bx,nE,1);
   
   		godunov->AdvectScalars(bx, dx, dt,
   									  D_DECL( area[0][mfi],     area[1][mfi],     area[2][mfi]),
   									  D_DECL( pnp_Ueff[0][mfi], pnp_Ueff[1][mfi], pnp_Ueff[2][mfi]),
   									  D_DECL( cflux[0],           cflux[1],           cflux[2]),
   									  D_DECL( edgstate[0],        edgstate[1],        edgstate[2]),
   									  neFab, 0, 1, force, 0, divu, 0, conv_ne[mfi], 0, advectionTypeLcl, state_bc, FPU, volume[mfi]);
      }
   }	

	conv_ne.mult(-1.0);

}

void PeleLM::compute_phiV_laplacian_term(const Real      &dt,
													  const MultiFab  &pnp_U,
													        MultiFab  *flux_phiV[AMREX_SPACEDIM],	
													        MultiFab  &lapl_phiV) {

// Get PhiV_alias with only 1 GC
	MultiFab PhiV_alias(pnp_U,amrex::make_alias,1,1);

// Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
	info.setMaxCoarseningLevel(0);
   MLPoisson phiV_poisson({geom}, {grids}, {dmap}, info);
   phiV_poisson.setMaxOrder(ef_PoissonMaxOrder);

// BC's. Use GC from S.
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
	ef_set_PoissonBC(mlmg_lobc, mlmg_hibc);
   phiV_poisson.setDomainBC(mlmg_lobc, mlmg_hibc);
   phiV_poisson.setLevelBC(0, &PhiV_alias);

// LinearSolver to get divergence. Use PhiV from NL vector pnp_U.
	MLMG mlmg(phiV_poisson);
	mlmg.apply({&lapl_phiV},{&PhiV_alias});

// Need the "fluxes" of Phi later on, get that now !
   std::array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(flux_phiV[0],flux_phiV[1],flux_phiV[2])};
	mlmg.getFluxes({fp},{&PhiV_alias});

// Rescale fluxes to get right stuff regardless of cartesian or r-Z
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
       flux_phiV[d]->mult(1.0/(geom.CellSize()[d]));   
		 flux_phiV[d]->FillBoundary(); 
	}

// TODO: the FillBoundary() only works on single level. Will have to to something else for AMR.
// 		For now crash if more than one level.
	BL_ASSERT(level==0);

}

void PeleLM::ef_calc_grad(const Real      &dt,
								  		  MultiFab  &field,
								        MultiFab  *grad_field[AMREX_SPACEDIM]) {	

// Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
	info.setMaxCoarseningLevel(0);
   MLPoisson poisson({geom}, {grids}, {dmap}, info);
   poisson.setMaxOrder(ef_PoissonMaxOrder);

// BC's. Use GC from S.
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
	ef_set_PoissonBC(mlmg_lobc, mlmg_hibc);
   poisson.setDomainBC(mlmg_lobc, mlmg_hibc);
   poisson.setLevelBC(0, &field);

// LinearSolver to get divergence.
	MLMG mlmg(poisson);
	MultiFab lapl(grids,dmap,1,1);
	mlmg.apply({&lapl},{&field});

// Need the "fluxes" to get the gradient.
   std::array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(grad_field[0],grad_field[1],grad_field[2])};
	mlmg.getFluxes({fp},{&field});

// Rescale fluxes to get right stuff regardless of cartesian or r-Z
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
       grad_field[d]->mult(1.0/(geom.CellSize()[d]));   
		 grad_field[d]->FillBoundary();
	}
		 
// TODO: the FillBoundary() only works on single level. Will have to to something else for AMR.
// 		For now crash if more than one level.
	BL_ASSERT(level==0);
}

void PeleLM::ef_GMRES_solve(const Real      &dt,
									 const Real      &norm_pnp_U,
									 const MultiFab  &pnp_U,
									 const MultiFab  &b_in,	
									       MultiFab  &x0) {

// Initial guess of dU = 0.0
	x0.setVal(0.0);

   MultiFab Ax(grids,dmap,2,0);
   MultiFab dx(grids,dmap,2,1);
   MultiFab x(grids,dmap,2,1);
   MultiFab r(grids,dmap,2,1);
	Vector<MultiFab> KspBase(ef_GMRES_size+1);				// Krylov ortho basis
	for (int d = 0; d <= ef_GMRES_size ; ++d) {
		KspBase[d].define(grids,dmap,2,1); 
	}

	Real beta = 0.0;													// Initial resisual
	Real rel_tol = 0.0;												// Target relative tolerance

   bool GMRES_converged = false;
   int GMRES_restart = 0;	
   do {																	// Outer restarted GMRES loop

	   Real H[ef_GMRES_size+1][ef_GMRES_size] = {0.0}; 	// Hessenberg mattrix
	   Real y[ef_GMRES_size+1] = {0.0};							// Solution vector
	   Real g[ef_GMRES_size+1] = {0.0};							// Residual
	   Real givens[ef_GMRES_size+1][2] = {0.0};				// Givens rotations

      ef_JtV(dt, norm_pnp_U, pnp_U, b_in, x0, Ax);			// Apply JtV on x0, useless if x0 == 0, ...
      MultiFab::LinComb(x,-1.0,Ax,0,1.0,b_in,0,0,2,0);	// Build residual		
      ef_applyPrecond(1000,x,r);    
      Real norm_r = ef_NL_norm(r);  							// Calc & store the initial residual norm
		if ( GMRES_restart == 0 ) {
			beta = norm_r;
			rel_tol = norm_r * ef_GMRES_reltol;
		}
	   amrex::Print() << " GMRES restart: " << GMRES_restart << " init residual: " << norm_r << "\n";

// 	Initialize KspBase with normalized residual		
		g[0] = norm_r;
		r.mult(1.0/norm_r);
		MultiFab::Copy(KspBase[0],r, 0 ,0, 2, 0);

//		showMFsub("pnp1D",b_in,stripBox,"1Dpnp_GMRESb_in",level);
//		showMFsub("pnp1D",r,stripBox,"1Dpnp_GMRESr0",level);
//		amrex::Abort("Step debugging");

		int k = 0;
		int k_end;
		Real norm_vec;
		Real norm_vec2;
		Real norm_Hlast;
		Real GS_corr;

		for ( int k = 0 ; k < ef_GMRES_size ; ++k ) {		// Inner restarted GMRES loop

			amrex::Print() << " --> GMRES ite: " << k << "\n"; 

			ef_JtV(dt, norm_pnp_U, pnp_U, b_in, KspBase[k],r);
			ef_applyPrecond(k,r,KspBase[k+1]);       
			norm_vec = ef_NL_norm(KspBase[k+1]);
//			amrex::Print() << " norm of KspBase k bef ortho : " << ef_NL_norm(KspBase[k]) << "\n";
//   	   showMF("pnp",KspBase[k+1],"pnp_KspBase_befortho",level,k+1);

//			Gram-Schmidt ortho.			
			for ( int row = 0; row <= k; ++row ) {
//				amrex::Print() << "     Ortho with GMRES vec row = " << row << "\n";
//				amrex::Print() << "     norm of KspBase vec row : " << ef_NL_norm(KspBase[row]) << "\n";
				H[row][k] = MultiFab::Dot(KspBase[k+1],0,KspBase[row],0,2,0); 
				GS_corr = - H[row][k];	
//				amrex::Print() << "     Dotprod with GMRES vec " << row << " before : " << H[row][k] << "\n";
				MultiFab::Saxpy(KspBase[k+1],GS_corr,KspBase[row],0,0,2,0);	
				Real Hcorr = MultiFab::Dot(KspBase[k+1],0,KspBase[row],0,2,0);
				if ( fabs(Hcorr) > 1.0e-15 ) {
					H[row][k] += Hcorr;
					GS_corr = - Hcorr;
					MultiFab::Saxpy(KspBase[k+1],GS_corr,KspBase[row],0,0,2,0);
				}
				amrex::Print() << "     Dotprod with GMRES vec " << row << " : " << MultiFab::Dot(KspBase[k+1],0,KspBase[row],0,2,0) << "\n";
			}
			norm_vec2 = ef_NL_norm(KspBase[k+1]);
//			amrex::Print() << "     GMRES base " << k+1 << " norm " << norm_vec2 << "\n"; 
			H[k+1][k] = norm_vec2;

//			TODO: Test for re-ortho.			
//			if ( norm_vec + 0.0001 * norm_vec2 == norm_vec ) {
//			}

			if ( norm_vec2 > 0.0 ) 
				KspBase[k+1].mult(1.0/norm_vec2);

//   	   showMF("pnp",KspBase[k+1],"pnp_KspBase_aftortho",level,k+1);

//       Givens rotation
         for ( int row = 0; row < k; ++row ) {
            Real v1 =   givens[row][0] * H[row][k] - givens[row][1] * H[row+1][k];
            Real v2 =   givens[row][1] * H[row][k] + givens[row][0] * H[row+1][k];
            H[row][k] = v1;   
            H[row+1][k] = v2;
			}

         norm_Hlast = std::sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
         if ( norm_Hlast > 0.0 ) { 
//			   amrex::Print() << " norm_Hlast: " << norm_Hlast << "\n";
				if (  H[k+1][k] == 0.0 ) {
					givens[k][0] = 1.0;
					givens[k][1] = 0.0;
				} else if ( std::abs(H[k+1][k]) > std::abs(H[k][k]) ) {
            	givens[k][0] =  H[k+1][k] / norm_Hlast;
            	givens[k][1] = -H[k][k] / norm_Hlast;
				} else {
            	givens[k][0] =  H[k][k] / norm_Hlast;
            	givens[k][1] = -H[k+1][k] / norm_Hlast;
				}			
            H[k][k] = givens[k][0] * H[k][k] - givens[k][1] * H[k+1][k];
            H[k+1][k] = 0.0;
            Real v1 = givens[k][0] * g[k]; //- givens[k][1] * g[k+1];
            Real v2 = givens[k][1] * g[k]; //+ givens[k][0] * g[k+1];
            g[k] = v1;
            g[k+1] = v2; 
         }

		   norm_r = fabs(g[k+1]);
			amrex::Print() << " restart GMRES residual: " << norm_r << "\n";

//			Exit conditions. k_end used here to construct the update.			
			if ( norm_r < rel_tol ) {
				GMRES_converged = true;
				k_end = k;
				break;
			}
			if ( k == ef_GMRES_size-1)
				k_end = ef_GMRES_size-1;
			
		}																			// Inner GMRES loop

//		Plot H mat		
//      for ( int i = 0; i <= ef_GMRES_size ; ++i ) {		
//			amrex::Print() << " - ";
//			for ( int j = 0; j < ef_GMRES_size ; ++j ) {
//				amrex::Print() << H[i][j] << " ";
//			}
//			amrex::Print() << "\n";
//		}

//		Solve H.y = g
		y[k_end] = g[k_end]/H[k_end][k_end];
		amrex::Print() << " y["<<k_end<<"] : " << y[k_end] << "\n";
		for ( int k = k_end-1; k >= 0; --k ) {
			Real sum_tmp = 0.0;
			for ( int j = k+1; j <= k_end; ++j ) {
			   sum_tmp += H[k][j] * y[j];
			}
			y[k] = ( g[k] - sum_tmp ) / H[k][k];
			amrex::Print() << " y["<<k<<"] : " << y[k] << "\n";
		} 

		amrex::Print() << " Finished restart GMRES in : " << k_end+1 << " iterations \n";

//		Compute solution update		
		r.setVal(0.0);
		for ( int i = 0; i <= k_end; ++i ) {
//			amrex::Print() << " Norm of " << i << " vec of KspBase: " << ef_NL_norm(KspBase[i]) << " \n";
			MultiFab::Saxpy(r,y[i],KspBase[i], 0, 0, 2, 0);
// 	   showMF("pnp",KspBase[i],"pnp_KspBase",level,i);
		}
		amrex::Print() << " Norm of solution update : " << ef_NL_norm(r) << " \n";

//		Update solution		
		MultiFab::Add(x0,r,0,0,2,0);
//   	showMF("pnp",x0,"pnp_x0_GMRES",level,GMRES_restart);
//		showMFsub("pnp1D",x0,stripBox,"1Dpnp_xendGMRES",level);

		GMRES_restart += 1;

   } while( ! GMRES_converged && GMRES_restart < ef_max_GMRES_rst );

//	amrex::Print() << " Finished with GMRES \n";

	KspBase.clear();

}

void PeleLM::ef_JtV(const Real      &dt,
	  	              const Real      &norm_pnp_U, 
			           const MultiFab  &pnp_U,	 
				        const MultiFab  &pnp_res,	 
				        const MultiFab  &v_in,
			                 MultiFab  &JtV) {	 

// Compute perturbation scaling
	Real normv = ef_NL_norm(v_in);
	if ( normv == 0.0 ) {
		JtV.setVal(0.0);
		return;
	}
   Real delta_pert = ef_lambda_jfnk * ( ef_lambda_jfnk + norm_pnp_U / normv );

// Get perturbed pnp vector	
   MultiFab U_pert(grids,dmap,2,Godunov::hypgrow());
//	MultiFab::LinComb(U_pert, 1.0, pnp_U, 0, delta_pert, v_in, 0,0,2,Godunov::hypgrow());
	MultiFab::Copy(U_pert, pnp_U, 0, 0, 2, Godunov::hypgrow());
	MultiFab::Saxpy(U_pert,delta_pert,v_in, 0, 0, 2 ,0);

// Get perturbed residual	
	MultiFab res_pert(grids,dmap,2,1);
	ef_NL_residual( dt, U_pert, res_pert );
   res_pert.mult(-1.0);
	
// Reuse U_pert MultiFab to store Jtv	
	MultiFab::LinComb(JtV,1.0,res_pert,0,-1.0,pnp_res,0,0,2,0);
	JtV.mult(-1.0/delta_pert);
}

void PeleLM::ef_setUpPrecond (const Real      &dt,
										const MultiFab  &pnp_U,
										MultiFab        *De_ec[BL_SPACEDIM]) {

	static bool first_time = true;

   if ( first_time ) {		
//    Some LPInfo stuff	
      LPInfo info;
      info.setAgglomeration(1);
      info.setConsolidation(1);
      info.setMetricTerm(false);

//    Define all three LinOps	
	   pnp_pc_Stilda.define({geom}, {grids}, {dmap}, info);
	   pnp_pc_diff.define({geom}, {grids}, {dmap}, info);
	   pnp_pc_drift.define({geom}, {grids}, {dmap}, info);
      pnp_pc_Stilda.setMaxOrder(ef_PoissonMaxOrder);
      pnp_pc_diff.setMaxOrder(ef_PoissonMaxOrder);
      pnp_pc_drift.setMaxOrder(ef_PoissonMaxOrder);

	   first_time = false;
	}

// Diff/Drift LinOp ( MLABecCecLaplacian )	
   {
//    Scalars includes the scaling of the Jacobian
      pnp_pc_diff.setScalars(-pnp_SUne/pnp_SFne, -dt*pnp_SUne/pnp_SFne, dt*pnp_SUne/pnp_SFne); 
//    Diagonal part	
	   MultiFab dummy(grids,dmap,1,1);
	   dummy.setVal(1.0);
	   pnp_pc_diff.setACoeffs(0, dummy);
//    Diffusion part	
	   std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{D_DECL(De_ec[0],De_ec[1],De_ec[2])};
	   pnp_pc_diff.setBCoeffs(0, bcoeffs);
//    Advection part
	   std::array<const MultiFab*,AMREX_SPACEDIM> ccoeffs{D_DECL(&pnp_Ueff[0],&pnp_Ueff[1],&pnp_Ueff[2])};
	   pnp_pc_diff.setCCoeffs(0, ccoeffs);
//		pnp_pc_diff.checkDiagonalDominance(0,0);
	}

// Stilda and Drift LinOp
// Start by getting the upwinded edge based nE*Kappa_e from cc.
   {
	   MultiFab nEtKappaE(grids,dmap,1,1);
	   MultiFab::Copy(nEtKappaE, pnp_U, 0, 0, 1, 1);
	   MultiFab::Multiply(nEtKappaE, kappaElec_cc, 0, 0, 1, 1);
		MultiFab** neKe_ec = 0;
      FluxBoxes edgeCoeff(this, 1, 1);
      neKe_ec = edgeCoeff.get();
      for (MFIter mfi(nEtKappaE); mfi.isValid(); ++mfi) {
	      const Box& box = mfi.validbox();

         for (int dir = 0; dir < BL_SPACEDIM; dir++) {
            FPLoc bc_lo = fpi_phys_loc(get_desc_lst()[State_Type].getBC(nE).lo(dir));
	    	   FPLoc bc_hi = fpi_phys_loc(get_desc_lst()[State_Type].getBC(nE).hi(dir));
	    	   const Box& ebox = mfi.nodaltilebox(dir);
	    	   center_to_edge_upwind_fancy((nEtKappaE)[mfi],(pnp_Ueff[dir])[mfi],(*neKe_ec[dir])[mfi],
	   	                                amrex::grow(box,amrex::BASISV(dir)), ebox, 0, 
	   	                                0, 1, geom.Domain(), bc_lo, bc_hi);
				(*neKe_ec[dir])[mfi].mult(pnp_SUphiV/pnp_SFne*dt,ebox,0,1);
         }
      }
      pnp_pc_drift.setScalars(0.0,1.0); 
		{
	      std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{D_DECL(neKe_ec[0],neKe_ec[1],neKe_ec[2])};
	      pnp_pc_drift.setBCoeffs(0, bcoeffs);
		}

		Real scaling_Laplacian = 0.0;
		getScalingLap(&scaling_Laplacian);

      for (int dir = 0; dir < BL_SPACEDIM; dir++) {
		   neKe_ec[dir]->mult(pnp_SUne/pnp_SFphiV);
	   	neKe_ec[dir]->plus(scaling_Laplacian*pnp_SUphiV/pnp_SFphiV,0,1);
      }
//		showMFsub("pnp1D",*neKe_ec[1],stripBox,"1DPC_StildaCoeff",level);
//		showMF("pnp",*neKe_ec[1],"pnp_PC_StildaCoeff",level);
      pnp_pc_Stilda.setScalars(0.0,-1.0); 
		{
	      std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{D_DECL(neKe_ec[0],neKe_ec[1],neKe_ec[2])};
	      pnp_pc_Stilda.setBCoeffs(0, bcoeffs);
		}
   }

}

void PeleLM::ef_applyPrecond ( const int      &GMRES_ite,
										 const MultiFab &v,
										       MultiFab &Pv) {

	MultiFab neComp_alias(v,amrex::make_alias,0,1);
	MultiFab phiVComp_alias(v,amrex::make_alias,1,1);
	MultiFab PneComp_alias(Pv,amrex::make_alias,0,1);
	MultiFab PphiVComp_alias(Pv,amrex::make_alias,1,1);
	PneComp_alias.setVal(0.0);
	PphiVComp_alias.setVal(0.0);

// Gather BC's for linear systems: all Neumann
// TODO: no idea whether this is right or not ...	
	std::array<LinOpBCType,AMREX_SPACEDIM> ne_lobc, ne_hibc;
   std::array<LinOpBCType,AMREX_SPACEDIM> phiV_lobc, phiV_hibc;
	ef_set_PoissonBC(phiV_lobc, phiV_hibc);
	ef_set_PCBC(ne_lobc, ne_hibc);

   pnp_pc_Stilda.setDomainBC(phiV_lobc, phiV_hibc);	
   pnp_pc_diff.setDomainBC(ne_lobc, ne_hibc);	
   pnp_pc_drift.setDomainBC(phiV_lobc, phiV_hibc);	

	pnp_pc_diff.setLevelBC(0, &PneComp_alias);
	pnp_pc_drift.setLevelBC(0, &PphiVComp_alias);
	pnp_pc_Stilda.setLevelBC(0, &PphiVComp_alias);

	MLMG mg_diff(pnp_pc_diff);
	MLMG mg_drift(pnp_pc_drift);
	MLMG mg_Stilda(pnp_pc_Stilda);
	mg_diff.setMaxFmgIter(20);
	mg_drift.setMaxFmgIter(20);
	mg_Stilda.setMaxFmgIter(20);
	mg_diff.setMaxIter(ef_PoissonMaxIter);
	mg_drift.setMaxIter(ef_PoissonMaxIter);
	mg_Stilda.setMaxIter(ef_PoissonMaxIter);
	mg_diff.setVerbose(ef_PoissonVerbose);
	mg_drift.setVerbose(ef_PoissonVerbose);
	mg_Stilda.setVerbose(ef_PoissonVerbose);

// Set-up solver tolerances
   Real S_tol     = ef_PC_MG_tol;
   Real S_tol_abs = neComp_alias.norm0() * ef_PC_MG_tol;

//     Most inner mat   
//     --                --
//     | [dtD-I]^-1     0 |
//     |                  |  
//     |       0        I |
//     --                -- 
	mg_diff.solve({&PneComp_alias}, {&neComp_alias}, S_tol, S_tol_abs);
	MultiFab::Copy(Pv,v,1,1,1,1);
//	showMFsub("pnp1D",Pv,stripBox,"1DPC_Pv_AftM1",level);

//     assembling mat
//     --       --
//     |  I    0 |
//     |         |  
//     | -Ie   I |
//     --       --
   MultiFab::Saxpy(PphiVComp_alias,pnp_SUne/pnp_SFphiV,PneComp_alias,0,0,1,1); 
//	showMFsub("pnp1D",Pv,stripBox,"1DPC_Pv_AftM2",level);


//     phiV estimate mat   
//     --         --
//     | I     0   |
//     |           |  
//     | 0   S*^-1 |
//     --         -- 
   MultiFab temp(grids,dmap,1,1);
   MultiFab temp2(grids,dmap,1,1);
	temp.setVal(0.0);
   S_tol_abs = PphiVComp_alias.norm0() * ef_PC_MG_tol;     
	mg_Stilda.solve({&temp},{&PphiVComp_alias}, S_tol, S_tol_abs);
	MultiFab::Copy(PphiVComp_alias, temp, 0, 0, 1, 1);
	temp.setVal(0.0);
//	showMFsub("pnp1D",Pv,stripBox,"1DPC_Pv_AftM3",level);

//     Final mat   
//     --                          --
//     | I       -[dtD - I]^-1 dtDr |
//     |                            |  
//     | 0                I         |
//     --                          --
   mg_drift.apply({&temp},{&PphiVComp_alias});
	S_tol_abs = temp.norm0() * ef_PC_MG_tol;
	temp2.setVal(0.0);
	mg_diff.solve({&temp2},{&temp}, S_tol, S_tol_abs);
	temp2.mult(-1.0);
	MultiFab::Add(PneComp_alias,temp2,0,0,1,1);
//	showMFsub("pnp1D",Pv,stripBox,"1DPC_Pv_AftM4",level);


//	amrex::Abort("In applyPrecond !");
}

void
PeleLM::center_to_edge_upwind_fancy (const FArrayBox& cfab,
                                     const FArrayBox& velefab,
                                     FArrayBox&       efab,
                                     const Box&       ccBox,
                                     const Box&       ebox,
                                     int              sComp,
                                     int              dComp,
                                     int              nComp,
                                     const Box&       domain,
                                     const FPLoc&     bc_lo,
                                     const FPLoc&     bc_hi)
{
  //const Box&      ebox = efab.box();
  const IndexType ixt  = ebox.ixType();

  BL_ASSERT(!(ixt.cellCentered()) && !(ixt.nodeCentered()));

  int dir = -1;
  for (int d = 0; d < BL_SPACEDIM; d++)
    if (ixt.test(d))
      dir = d;

  BL_ASSERT(amrex::grow(ccBox,-amrex::BASISV(dir)).contains(amrex::enclosedCells(ebox)));
  BL_ASSERT(sComp+nComp <= cfab.nComp() && dComp+nComp <= efab.nComp());
  //
  // Exclude unnecessary cc->ec calcs
  //
  Box ccVBox = ccBox;
  if (bc_lo!=HT_Center)
    ccVBox.setSmall(dir,std::max(domain.smallEnd(dir),ccVBox.smallEnd(dir)));
  if (bc_hi!=HT_Center)
    ccVBox.setBig(dir,std::min(domain.bigEnd(dir),ccVBox.bigEnd(dir)));
  //
  // Shift cell-centered data to edges
  //
  const int isharm = def_harm_avg_cen2edge?1:0;
  cen2edgup(ebox.loVect(),ebox.hiVect(),
            ARLIM(cfab.loVect()),ARLIM(cfab.hiVect()),cfab.dataPtr(sComp),
            ARLIM(velefab.loVect()),ARLIM(velefab.hiVect()),velefab.dataPtr(0),
            ARLIM(efab.loVect()),ARLIM(efab.hiVect()),efab.dataPtr(dComp),
            &nComp, &dir, &isharm);
  //
  // Fix boundary...i.e. fill-patched data in cfab REALLY lives on edges
  //
  if ( !(domain.contains(ccBox)) )
  {
    if (bc_lo==HT_Edge)
    {
      BoxList gCells = amrex::boxDiff(ccBox,domain);
      if (gCells.ok())
      {
        const int inc = +1;
        FArrayBox ovlpFab;
        for (BoxList::const_iterator bli = gCells.begin(), end = gCells.end();
             bli != end;
             ++bli)
        {
          if (bc_lo == HT_Edge)
          {
            ovlpFab.resize(*bli,nComp);
            ovlpFab.copy(cfab,sComp,0,nComp);
            ovlpFab.shiftHalf(dir,inc);
            efab.copy(ovlpFab,0,dComp,nComp);
          }
        }
      }
    }
    if (bc_hi==HT_Edge)
    {
      BoxList gCells = amrex::boxDiff(ccBox,domain);
      if (gCells.ok())
      {
        const int inc = -1;
        FArrayBox ovlpFab;
        for (BoxList::const_iterator bli = gCells.begin(), end = gCells.end();
             bli != end;
             ++bli)
        {
          if (bc_hi == HT_Edge)
          {
            ovlpFab.resize(*bli,nComp);
            ovlpFab.copy(cfab,sComp,0,nComp);
            ovlpFab.shiftHalf(dir,inc);
            efab.copy(ovlpFab,0,dComp,nComp);
          }
        }
      }
    }
  }
}    

// Setup BC conditions for diffusion operator on nE. Directly copied from the diffusion one ...
void PeleLM::ef_set_neBC(std::array<LinOpBCType,AMREX_SPACEDIM> &diff_lobc,
                         std::array<LinOpBCType,AMREX_SPACEDIM> &diff_hibc) {

    const BCRec& bc = get_desc_lst()[State_Type].getBC(nE);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (Geom().isPeriodic(idim))
        {
            diff_lobc[idim] = diff_hibc[idim] = LinOpBCType::Periodic;
        }
        else
        {
            int pbc = bc.lo(idim);
            if (pbc == EXT_DIR)
            {
                diff_lobc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      || 
                     pbc == REFLECT_EVEN)
            {
                diff_lobc[idim] = LinOpBCType::Neumann;
            }
            else if (pbc == REFLECT_ODD)
            {
                diff_lobc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                diff_lobc[idim] = LinOpBCType::bogus;
            }

            pbc = bc.hi(idim);
            if (pbc == EXT_DIR)
            {
                diff_hibc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      ||
                     pbc == REFLECT_EVEN)
            {
                diff_hibc[idim] = LinOpBCType::Neumann;
				} 
            else if (pbc == REFLECT_ODD)
            {
                diff_hibc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                diff_hibc[idim] = LinOpBCType::bogus;
            }
        }
    }
}

// Setup BC conditions for linear Poisson solve on PhiV. Directly copied from the diffusion one ...
void PeleLM::ef_set_PoissonBC(std::array<LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                              std::array<LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc) {

    const BCRec& bc = get_desc_lst()[State_Type].getBC(PhiV);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (Geom().isPeriodic(idim))
        {
            mlmg_lobc[idim] = mlmg_hibc[idim] = LinOpBCType::Periodic;
        }
        else
        {
            int pbc = bc.lo(idim);
            if (pbc == EXT_DIR)
            {
                mlmg_lobc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      || 
                     pbc == REFLECT_EVEN)
            {
                mlmg_lobc[idim] = LinOpBCType::Neumann;
            }
            else if (pbc == REFLECT_ODD)
            {
                mlmg_lobc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                mlmg_lobc[idim] = LinOpBCType::bogus;
            }

            pbc = bc.hi(idim);
            if (pbc == EXT_DIR)
            {
                mlmg_hibc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      || 
                     pbc == REFLECT_EVEN)
            {
                mlmg_hibc[idim] = LinOpBCType::Neumann;
				} 
            else if (pbc == REFLECT_ODD)
            {
                mlmg_hibc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                mlmg_hibc[idim] = LinOpBCType::bogus;
            }
        }
    }
}

void PeleLM::ef_set_PCBC(std::array<LinOpBCType,AMREX_SPACEDIM> &diff_lobc,
                         std::array<LinOpBCType,AMREX_SPACEDIM> &diff_hibc) {

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (Geom().isPeriodic(idim))
        {
            diff_lobc[idim] = diff_hibc[idim] = LinOpBCType::Periodic;
        }
        else
        {
			   diff_lobc[idim] = diff_hibc[idim] = LinOpBCType::Neumann;
		  }
    }
}
