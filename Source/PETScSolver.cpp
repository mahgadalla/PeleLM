#include <petscksp.h>
#include <PETScSolver.H>
#include <PETScSolver_F.H>
#include <AMReX_VisMF.H>

namespace amrex {

PETScSolver::PETScSolver (const BoxArray& grids, const DistributionMapping& dmap,
                          const Geometry& geom_, MPI_Comm comm_)
   : comm(comm_),
     geom(geom_)
{
   //amrex::Print() << " PETScSolver::constructor \n";

   verbose = 1;

   m_maxorder = 2;

   int ncomp = 1;
   int ngrow = 0;

   for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      BoxArray edge_boxes(grids);
      edge_boxes.surroundingNodes(i);
      De_coefs[i].define(edge_boxes, dmap, ncomp, ngrow);
      effvel_coefs[i].define(edge_boxes, dmap, ncomp, ngrow);
      neKeupwind_coefs[i].define(edge_boxes, dmap, ncomp, ngrow);
      De_coefs[i].setVal(0.0);
      effvel_coefs[i].setVal(0.0);
      neKeupwind_coefs[i].setVal(0.0);
   }

   diaginv.define(grids,dmap,2,ngrow);  // We've got diag for ne cells and phiV cells

   PETSC_COMM_WORLD = comm; 

   const char* option_char;
   const char* value_char;

   option_char = "-pc_type";  
   value_char = "lu";
   PetscOptionsSetValue(NULL, option_char, value_char);
   option_char = "-pc_factor_mat_solver_type";
   //value_char = "superlu_dist";
   value_char = "superlu";
   PetscOptionsSetValue(NULL, option_char, value_char);
   option_char = "-ksp_type";
   value_char = "preonly";
   PetscOptionsSetValue(NULL, option_char, value_char);
   //option_char = "-start_in_debugger";
   //value_char = " ";
   //PetscOptionsSetValue(NULL, option_char, value_char);
   //option_char = "-mat_superlu_ilu_droptol";
   //value_char = "1.e-70";
   //PetscOptionsSetValue(NULL, option_char, value_char);

   PetscInitialize(0, 0, 0, 0);
}

PETScSolver::~PETScSolver()
{
   //amrex::Print() << " PETScSolver::destructor \n";

   MatDestroy(&A);
   A = nullptr;

   VecDestroy(&b);
   b = nullptr;

   VecDestroy(&x);
   x = nullptr;

   KSPDestroy(&solver);
   solver = nullptr;

   PetscFinalize();
}

void
PETScSolver::setDriftFaceCoeffs(const Array<const MultiFab*, AMREX_SPACEDIM>& neKe_ec_upwind)
{
   for (int idim=0; idim < AMREX_SPACEDIM; idim++) {
      MultiFab::Copy(neKeupwind_coefs[idim], *neKe_ec_upwind[idim], 0, 0, 1, 0);
   }   
}

void
PETScSolver::setDiffFaceCoeffs(const Array<const MultiFab*, AMREX_SPACEDIM>& De_ec)
{
   for (int idim=0; idim < AMREX_SPACEDIM; idim++) {
      MultiFab::Copy(De_coefs[idim], *De_ec[idim], 0, 0, 1, 0);
   }   
}

void
PETScSolver::setUeffFaceCoeffs(const Array<const MultiFab*, AMREX_SPACEDIM>& Ueff_ec_upwind)
{
   for (int idim=0; idim < AMREX_SPACEDIM; idim++) {
      MultiFab::Copy(effvel_coefs[idim], *Ueff_ec_upwind[idim], 0, 0, 1, 0);
   }   
}

void
PETScSolver::prepareSolver()
{
   //amrex::Print() << " PETScSolver::prepareSolver \n ";

   int num_procs, myid;
   MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);
   MPI_Comm_rank(PETSC_COMM_WORLD, &myid);

   //std::cout << " My myid " << myid << " amongst " << num_procs << " \n ";

   const BoxArray& ba = diaginv.boxArray();
   const DistributionMapping& dm = diaginv.DistributionMap();

// Set-up the necessary data structure to handle PETSc
   ncells_grid.define(ba,dm);
   cell_id_ne.define(ba,dm,1,1);
   cell_id_phiV.define(ba,dm,1,1);
   cell_id_ne_vec.define(ba,dm);
   cell_id_phiV_vec.define(ba,dm);

// Let's make an inventory of the cell we've got   
// After this MFiter, both cell_id_ne & cell_id_phiV contains int from 0 to box.numPts()-1 for each box in the MF
   PetscInt ncells_proc = 0;  // My amount of cells

   {
      BaseFab<PetscInt> ifab;   
      for (MFIter mfi(cell_id_ne); mfi.isValid(); ++mfi)
      {
         const Box& bx = mfi.validbox();
         BaseFab<PetscInt>& neid_fab = cell_id_ne[mfi];
         BaseFab<PetscInt>& phiVid_fab = cell_id_phiV[mfi];
         neid_fab.setVal(std::numeric_limits<PetscInt>::lowest());
         phiVid_fab.setVal(std::numeric_limits<PetscInt>::lowest());
         long npts = bx.numPts();
         ncells_grid[mfi] = npts;
         ncells_proc += npts;
         ifab.resize(bx);
         PetscInt* p = ifab.dataPtr();
         for (long i = 0; i < npts; ++i) {
             *p++ = i; 
         }   
         neid_fab.copy(ifab,bx);
         phiVid_fab.copy(ifab,bx);
      }
	}
   //std::cout << "["<< myid << "]  ncells_proc:  " << ncells_proc << " \n ";

   Vector<PetscInt> ncells_allprocs(num_procs);                                                                                                         
   MPI_Allgather(&ncells_proc, sizeof(PetscInt), MPI_CHAR,
                 ncells_allprocs.data(), sizeof(PetscInt), MPI_CHAR,
                 PETSC_COMM_WORLD);
   PetscInt proc_begin = 0;    // Index where I start (sum of ncells_proc for ranks < mine)
   for (int i = 0; i < myid; ++i) {
       proc_begin += ncells_allprocs[i];
   }
   PetscInt ncells_world = 0;  // Total number of cells across the procs
   for (auto i : ncells_allprocs) {
       ncells_world += i;
   }
   //std::cout << "["<< myid << "]  ncells_world :  " << ncells_world << " \n ";
   //std::cout << "["<< myid << "]  I begin at :  " << proc_begin << " \n ";

   LayoutData<PetscInt> offset_ne(ba,dm);     // Where does each of my boxes start (offset) for nE
   LayoutData<PetscInt> offset_phiV(ba,dm);   // Where does each of my boxes start (offset) for phiV
   PetscInt proc_end = proc_begin;
   for (MFIter mfi(ncells_grid); mfi.isValid(); ++mfi)
   {   
       offset_ne[mfi] = proc_end;
       offset_phiV[mfi] = proc_end + ncells_world;
       proc_end += ncells_grid[mfi];
		 //std::cout << "["<< myid << "]  box(" << mfi.index() <<") ne offset: " << offset_ne[mfi] << " \n ";
		 //std::cout << "["<< myid << "]  box(" << mfi.index() <<") phiV offset: " << offset_phiV[mfi] << " \n ";
   }   

// After this MFiter, cell_id_ne now contains int from 0 to box.numPts()-1 + offset for each box in the MF
// while cell_id_phiV is further offsetted by the overall number of cells
   for (MFIter mfi(cell_id_ne); mfi.isValid(); ++mfi)
   {
       cell_id_ne[mfi].plus(offset_ne[mfi], mfi.tilebox());
       cell_id_phiV[mfi].plus(offset_phiV[mfi], mfi.tilebox());
   }    

   cell_id_ne.FillBoundary(geom.periodicity());
   cell_id_phiV.FillBoundary(geom.periodicity());

// Build the Jacobian matrix for PETSc   
   MatCreate(PETSC_COMM_WORLD, &A);

   if ( num_procs == 1 ) {
//    The size of A is twice the total number of cells (since ne + phiV)   
      MatSetType(A, MATSEQAIJ);
      MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, ncells_world*2, ncells_world*2);
      PetscInt d_nz = (2*AMREX_SPACEDIM+1)*2;
      PetscInt o_nz = (2*AMREX_SPACEDIM+1)*1;
      MatSeqAIJSetPreallocation(A,d_nz,NULL);
      MatSetUp(A);
      MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
   } else {
//    The size of A is twice the total number of cells (since ne + phiV)   
      MatSetType(A, MATMPIAIJ);
      MatSetSizes(A, ncells_proc*2, ncells_proc*2, ncells_world*2, ncells_world*2);
      PetscInt d_nz = (2*AMREX_SPACEDIM+1)*2;
      PetscInt o_nz = (2*AMREX_SPACEDIM+1)*1;
      MatMPIAIJSetPreallocation(A, d_nz, NULL, o_nz, NULL );
      MatSetUp(A);
      MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
   }

// Fill the matrix
   const Real* dx = geom.CellSize();
   FArrayBox rfab;
   BaseFab<PetscInt> ifab;
   BaseFab<PetscInt> ifab2;
   for (MFIter mfi(diaginv); mfi.isValid(); ++mfi)
   {   
      //std::cout << "["<< myid << "]  Box " << mfi.index() << " \n ";
      const Box& bx = mfi.validbox();

      Array<int,AMREX_SPACEDIM*2> bctype_ne;
      Array<int,AMREX_SPACEDIM*2> bctype_phiV;
      Array<Real,AMREX_SPACEDIM*2> bcl_ne;
      Array<Real,AMREX_SPACEDIM*2> bcl_phiV;
      const Vector< Vector<BoundCond> > & bcs_ne_i = m_bndry_ne->bndryConds(mfi);
      const BndryData::RealTuple        & bcl_ne_i = m_bndry_ne->bndryLocs(mfi);
      const Vector< Vector<BoundCond> > & bcs_phiV_i = m_bndry_phiV->bndryConds(mfi);
      const BndryData::RealTuple        & bcl_phiV_i = m_bndry_phiV->bndryLocs(mfi);
      for (OrientationIter oit; oit; oit++) {
         int cdir(oit());
         bctype_ne[cdir] = bcs_ne_i[cdir][0];
         bcl_ne[cdir]  = bcl_ne_i[cdir];
         bctype_phiV[cdir] = bcs_phiV_i[cdir][0];
         bcl_phiV[cdir]  = bcl_phiV_i[cdir];
      }

      const PetscInt max_stencil_size = 2*(2*2 + 1);  // 
      const PetscInt nrows = ncells_grid[mfi];                 // Number of cells in the box

      ifab.resize(bx,2);                                       // Nb of entries / row
      ifab2.resize(bx,max_stencil_size*2);                     // Col of each entry / per
      rfab.resize(bx,max_stencil_size*2);                      // Entries
      cell_id_ne_vec[mfi].resize(nrows);
      cell_id_phiV_vec[mfi].resize(nrows);

      PetscInt* rowsnE    = cell_id_ne_vec[mfi].data();
      PetscInt* rowsphiV  = cell_id_phiV_vec[mfi].data();
      PetscInt* ncolsnE   = ifab.dataPtr(0);
      PetscInt* ncolsphiV = ifab.dataPtr(1);
      PetscInt* colsnE    = ifab2.dataPtr(0);
      PetscInt* colsphiV  = ifab2.dataPtr(max_stencil_size);
      Real*     matnE     = rfab.dataPtr(0);
      Real*     matphiV   = rfab.dataPtr(max_stencil_size);

      petsc_fillmatbox(BL_TO_FORTRAN_BOX(bx),
                       &nrows,
                       ncolsnE, ncolsphiV,
                       rowsnE, rowsphiV, 
                       colsnE, matnE,
                       colsphiV, matphiV,
                       BL_TO_FORTRAN_ANYD(cell_id_ne[mfi]),
                       BL_TO_FORTRAN_ANYD(cell_id_phiV[mfi]),
                       &(offset_ne[mfi]), &(offset_phiV[mfi]),
                       BL_TO_FORTRAN_ANYD(diaginv[mfi]),
                       AMREX_D_DECL(BL_TO_FORTRAN_ANYD(De_coefs[0][mfi]),
                                    BL_TO_FORTRAN_ANYD(De_coefs[1][mfi]),   
                                    BL_TO_FORTRAN_ANYD(De_coefs[2][mfi])),
                       AMREX_D_DECL(BL_TO_FORTRAN_ANYD(effvel_coefs[0][mfi]),
                                    BL_TO_FORTRAN_ANYD(effvel_coefs[1][mfi]),   
                                    BL_TO_FORTRAN_ANYD(effvel_coefs[2][mfi])),
                       AMREX_D_DECL(BL_TO_FORTRAN_ANYD(neKeupwind_coefs[0][mfi]),
                                    BL_TO_FORTRAN_ANYD(neKeupwind_coefs[1][mfi]),   
                                    BL_TO_FORTRAN_ANYD(neKeupwind_coefs[2][mfi])),
                       &lapl_fac,
                       bctype_ne.data(), bcl_ne.data(),
                       bctype_phiV.data(), bcl_phiV.data(),
                       &s_dtDiffI, &s_dtDr, &s_Ie, &s_L, dx, &m_dt);

      //Load in by row! 
      int matidnE = 0;  
      int matidphiV = 0;  
      for (int rit = 0; rit < nrows; ++rit)
      {   
		    //std::cout << "["<< myid << "] nE (row=" << rowsnE[rit] <<",col=" << colsnE[matidnE] << ") value : "<< matnE[matidnE] << " \n ";
		    //std::cout << "["<< myid << "] phiV (row=" << rowsphiV[rit] <<",col=" << colsphiV[matidphiV] << ") value : "<< matphiV[matidphiV] << " \n ";
          MatSetValues(A, 1, &rowsnE[rit], ncolsnE[rit], &colsnE[matidnE], &matnE[matidnE], INSERT_VALUES);
          MatSetValues(A, 1, &rowsphiV[rit], ncolsphiV[rit], &colsphiV[matidphiV], &matphiV[matidphiV], INSERT_VALUES);
          matidnE += ncolsnE[rit];
          matidphiV += ncolsphiV[rit];
      }   
      
	}

   //amrex::Print() << " PETScSolver::prepareSolver done with petsc_fillmatbox MFIter \n";

   MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

//   MatView(A,PETSC_VIEWER_STDOUT_SELF);

//   Abort("Dumb");

   // create solver
   KSPCreate(PETSC_COMM_WORLD, &solver);
   KSPSetOperators(solver, A, A);

   // Set up preconditioner
   PC pc; 
   KSPGetPC(solver, &pc);

   PCSetFromOptions(pc);
   KSPSetFromOptions(solver);

   // Classic AMG
   //PCSetType(pc, PCGAMG);
   //PCGAMGSetType(pc, PCGAMGAGG);
   //PCGAMGSetNSmooths(pc,1); 
        
   // create b & x
   VecCreateMPI(PETSC_COMM_WORLD, ncells_proc*2, ncells_world*2, &x);
   VecDuplicate(x, &b);

}

void
PETScSolver::solve (MultiFab& soln, const MultiFab& rhs, Real rel_tol, Real abs_tol, 
                    const BndryData& bndry_ne, const BndryData& bndry_phiV, int max_iter, int max_bndry_order)
{
    BL_PROFILE("PETScSolver::solve()");

    if (m_bndry_ne != &bndry_ne || solver == nullptr || m_maxorder != max_bndry_order)
    {   
        m_bndry_ne = &bndry_ne;
        m_bndry_phiV = &bndry_phiV;
        m_maxorder = max_bndry_order;
        m_factory = &(rhs.Factory());
        prepareSolver();
    }   
    else
    {   
        m_factory = &(rhs.Factory());
    }   

    loadVectors(soln, rhs);
    //amrex::Print() << " PETScSolver::solve : filling sol & rhs vectors done \n";
    //  
    VecAssemblyBegin(x); 
    VecAssemblyEnd(x); 
    //  
    VecAssemblyBegin(b); 
    VecAssemblyEnd(b); 
    //amrex::Print() << " PETScSolver::solve : building sol & rhs vectors done \n";
    KSPSetTolerances(solver, rel_tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iter);
    KSPSolve(solver, b, x); 
    if (verbose >= 2)
    {   
        PetscInt niters;
        Real res;
        KSPGetIterationNumber(solver, &niters);
        KSPGetResidualNorm(solver, &res);
        amrex::Print() <<"\n" <<  niters << " PETSc Iterations, Residual Norm " << res << std::endl;
    }   

    getSolution(soln);
}

void
PETScSolver::loadVectors (MultiFab& soln, const MultiFab& rhs)
{
    BL_PROFILE("PETScSolver::loadVectors()");

    soln.setVal(0.0);

    FArrayBox rhsnEfab, rhsphiVfab;
    for (MFIter mfi(soln); mfi.isValid(); ++mfi)
    {   
        const Box& bx = mfi.validbox();
        const PetscInt nrows = ncells_grid[mfi];

	     const auto& neSolFab = soln[mfi].dataPtr(0);
	     const auto& phiVSolFab = soln[mfi].dataPtr(1);		

        // soln has been set to zero.
        VecSetValues(x, nrows, cell_id_ne_vec[mfi].data(), neSolFab, INSERT_VALUES); 
        VecSetValues(x, nrows, cell_id_phiV_vec[mfi].data(), phiVSolFab, INSERT_VALUES); 

        // Scale RHS
        rhsnEfab.resize(bx);
        rhsphiVfab.resize(bx);
        rhsnEfab.copy(rhs[mfi],bx,0,bx,0,1);
        rhsnEfab.mult(diaginv[mfi],bx,0,0,1);
        rhsphiVfab.copy(rhs[mfi],bx,1,bx,0,1);
        rhsphiVfab.mult(diaginv[mfi],bx,1,0,1);

        FArrayBox *bnEfab, *bphiVfab;
        {   
           bnEfab = &rhsnEfab;
           bphiVfab = &rhsphiVfab;
           VecSetValues(b, nrows, cell_id_ne_vec[mfi].data(), bnEfab->dataPtr(), INSERT_VALUES); 
           VecSetValues(b, nrows, cell_id_phiV_vec[mfi].data(), bphiVfab->dataPtr(), INSERT_VALUES); 
        }   
    }   
}

void
PETScSolver::getSolution (MultiFab& soln)
{
    FArrayBox rfab_nE, rfab_phiV;
    for (MFIter mfi(soln); mfi.isValid(); ++mfi)
    {   
        const Box& bx = mfi.validbox();
        const PetscInt nrows = ncells_grid[mfi];

        FArrayBox *xfab_ne, *xfab_phiV;
        if (soln.nGrow() == 0) {
            xfab_ne = &soln[mfi];
            xfab_phiV = &soln[mfi];
            VecGetValues(x, nrows, cell_id_ne_vec[mfi].data(), xfab_ne->dataPtr(0)); 
            VecGetValues(x, nrows, cell_id_phiV_vec[mfi].data(), xfab_phiV->dataPtr(1)); 
        } else {
            xfab_ne = &rfab_nE;
            xfab_phiV = &rfab_phiV;
            xfab_ne->resize(bx);
            xfab_phiV->resize(bx);
            VecGetValues(x, nrows, cell_id_ne_vec[mfi].data(), xfab_ne->dataPtr(0)); 
            VecGetValues(x, nrows, cell_id_phiV_vec[mfi].data(), xfab_phiV->dataPtr(0)); 
            soln[mfi].copy(*xfab_ne,bx,0,bx,0,1);
            soln[mfi].copy(*xfab_phiV,bx,0,bx,1,1);
        }   

    }   
}
}
