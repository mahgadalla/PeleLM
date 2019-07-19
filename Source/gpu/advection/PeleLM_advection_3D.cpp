#include "PeleLM_advection_3D.H"
#include "PeleLM_advection.H" 
#include "PeleLM.H"

using namespace amrex;

void PeleLM_AdvectScalars(const Box bx, const Real del[], 
                          const Real dt, const Array4<const Real> &Ax, 
                          const Array4<const Real> &Ay, const Array4<const Real> &Az, 
                          const Array4<const Real> &umac, const Array4<const Real> &vmac, 
                          const Array4<const Real> &wmac, const Array4<Real> &xflx, 
                          const Array4<Real> &yflx, const Array4<Real> &zflx, 
                          const Array4<Real> &xstate, const Array4<Real> &ystate, 
                          const Array4<Real> &zstate, const Array4<const Real> &s,
                          const Array4<const Real> &forces, const Array4<const Real> &divu, 
                          const Array4<Real> &aofs,
                          const amrex::Vector<AdvectionForm> &advectionType, 
                          const GpuArray<BCRec, NCOMP> &BCs,
                          const Array4<const Real> &V)
{
    int use_conserv_diff[NCOMP]; 
    for( int i = 0; i < NCOMP; ++i) 
        use_conserv_diff[i] = (advectionType[i] == 0) ? 1 : 0; 

    const int*  domain_lo = DefaultGeometry().Domain().loVect();
    const int*  domain_hi = DefaultGeometry().Domain().hiVect();

    PeleLM_estate_fpu(bx, s, forces, divu, umac, vmac, wmac, xstate, ystate, zstate, dt, 
                     del, BCs, domain_lo, domain_hi, use_conserv_diff); 

//ComputeAofs
    const Box fxbx = surroundingNodes(bx, 0); 
    const Box fybx = surroundingNodes(bx, 1); 
    const Box fzbx = surroundingNodes(bx, 2); 

    AMREX_PARALLEL_FOR_4D(fxbx, NCOMP, i, j, k, n, { 
        xflx(i,j,k,n) = xstate(i,j,k,n)*umac(i,j,k)*Ax(i,j,k);
    }); 
    AMREX_PARALLEL_FOR_4D(fybx, NCOMP, i, j, k, n, { 
        yflx(i,j,k,n) = ystate(i,j,k,n)*vmac(i,j,k)*Ay(i,j,k); 
    });
    AMREX_PARALLEL_FOR_4D(fzbx, NCOMP, i, j, k, n, { 
        zflx(i,j,k,n) = zstate(i,j,k,n)*wmac(i,j,k)*Az(i,j,k); 
    });

// Advective tendency that depends on the flux divergence 
    AMREX_PARALLEL_FOR_4D(bx, NCOMP, i, j, k, n, { 
        amrex::Real fluxdiv = ((xflx(i+1,j,k,n) - xflx(i,j,k,n))
                            +  (yflx(i,j+1,k,n) - yflx(i,j,k,n))
                            +  (zflx(i,j,k+1,n) - zflx(i,j,k,n)))/V(i,j,k);  

        if(use_conserv_diff[n]==0){
            amrex::Real divux = Ax(i+1,j,k)*umac(i+1,j,k) - 
                                Ax(i  ,j,k)*umac(i  ,j,k); 
            amrex::Real divuy = Ay(i,j+1,k)*vmac(i,j+1,k) - 
                                Ay(i,j  ,k)*vmac(i,j  ,k); 
            amrex::Real divuz = Az(i,j,k+1)*wmac(i,j,k+1) - 
                                Az(i,j,k  )*wmac(i,j,k  ); 
            aofs(i,j,k,n+RHO) = fluxdiv + (-divux*0.5*(xstate(i+1,j,k,n) - xstate(i,j,k,n))
                                 -divuy*0.5*(ystate(i,j+1,k,n) - ystate(i,j,k,n))
                                 -divuz*0.5*(zstate(i,j,k+1,n) - zstate(i,j,k,n)))/V(i,j,k);
            }
        else{
            aofs(i,j,k,n+RHO) = fluxdiv;
        }
    }); 
}

 
void PeleLM_estate_fpu(const amrex::Box bx,
                       const amrex::Array4<const amrex::Real> &s, 
                       const amrex::Array4<const amrex::Real> &tf, 
                       const amrex::Array4<const amrex::Real> &divu, 
                       const amrex::Array4<const amrex::Real> &umac, 
                       const amrex::Array4<const amrex::Real> &vmac, 
                       const amrex::Array4<const amrex::Real> &wmac, 
                       const amrex::Array4<amrex::Real> &xstate, 
                       const amrex::Array4<amrex::Real> &ystate, 
                       const amrex::Array4<amrex::Real> &zstate, 
                       const amrex::Real dt, const amrex::Real del[],
                       const GpuArray<BCRec, NCOMP> &BCs, const int domlo[],
                       const int domhi[], const int iconserv[]) 
{    
    auto const g2bx = amrex::grow(bx, 2); 
    auto const g1bx = amrex::grow(bx, 1); 

    const amrex::Real dx = del[0]; 
    const amrex::Real dy = del[1]; 
    const amrex::Real dz = del[2];
 
    const int domlox = domlo[0]; 
    const int domloy = domlo[1]; 
    const int domloz = domlo[2]; 
    
    const int domhix = domhi[0]; 
    const int domhiy = domhi[1]; 
    const int domhiz = domhi[2]; 
   
    auto const gxbx = amrex::grow(g2bx,0, 1); 
    FArrayBox Imxf(gxbx, NCOMP); 
    FArrayBox Ipxf(gxbx, NCOMP);
    Elixir Imxeli = Imxf.elixir(); 
    Elixir Ipxeli = Ipxf.elixir(); 
    auto const Imx = Imxf.array(); 
    auto const Ipx = Ipxf.array(); 

    auto const gybx = amrex::grow(g2bx,1, 1); 
    FArrayBox Imyf(gybx, NCOMP); 
    FArrayBox Ipyf(gybx, NCOMP);
    Elixir Imyeli = Imyf.elixir(); 
    Elixir Ipyeli = Ipyf.elixir(); 
    auto const Imy = Imyf.array(); 
    auto const Ipy = Ipyf.array(); 

   
    auto const gzbx = amrex::grow(g2bx,2, 1); 
    FArrayBox Imzf(gzbx, NCOMP); 
    FArrayBox Ipzf(gzbx, NCOMP);
    Elixir Imzeli = Imzf.elixir(); 
    Elixir Ipzeli = Ipzf.elixir(); 
    auto const Imz = Imzf.array(); 
    auto const Ipz = Ipzf.array(); 

/* Temporary Edge States */ 
    FArrayBox xedgef(g2bx, NCOMP); 
    FArrayBox yedgef(g2bx, NCOMP); 
    FArrayBox zedgef(g2bx, NCOMP); 

    Elixir xedeli = xedgef.elixir(); 
    Elixir yedeli = yedgef.elixir(); 
    Elixir zedeli = zedgef.elixir(); 

    auto const xedge = xedgef.array(); 
    auto const yedge = yedgef.array(); 
    auto const zedge = zedgef.array(); 

    auto const xgbx = surroundingNodes(g1bx, 0);
    auto const ygbx = surroundingNodes(g1bx, 1); 
    auto const zgbx = surroundingNodes(g1bx, 2); 

    /* Use PPM to generate Im and Ip */
 
    AMREX_PARALLEL_FOR_4D (g1bx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n];
        PeleLM_ppm(i, j, k, n, dt, dx, s, umac, Imx, Ipx, bc, bx.loVect()[0], bx.hiVect()[0], 0);
    });

    AMREX_PARALLEL_FOR_4D (g1bx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n];
        PeleLM_ppm(i, j, k, n, dt, dy, s, vmac, Imy, Ipy, bc, bx.loVect()[1], bx.hiVect()[1], 1);
    });

    AMREX_PARALLEL_FOR_4D (g1bx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n];
        PeleLM_ppm(i, j, k, n, dt, dz, s, wmac, Imz, Ipz, bc, bx.loVect()[2], bx.hiVect()[2], 2);
    }); 


    FArrayBox xlf(xgbx, NCOMP); 
    FArrayBox xhf(xgbx, NCOMP);
    Elixir xleli = xlf.elixir(); 
    Elixir xheli = xhf.elixir(); 
    auto const xlo = xlf.array(); 
    auto const xhi = xhf.array(); 

    FArrayBox ylf(ygbx, NCOMP); 
    FArrayBox yhf(ygbx, NCOMP);
    Elixir yleli = ylf.elixir(); 
    Elixir yheli = yhf.elixir(); 
    auto const ylo = ylf.array(); 
    auto const yhi = yhf.array(); 

   
    FArrayBox zlf(zgbx, NCOMP); 
    FArrayBox zhf(zgbx, NCOMP);
    Elixir zleli = zlf.elixir(); 
    Elixir zheli = zhf.elixir(); 
    auto const zlo = zlf.array(); 
    auto const zhi = zhf.array();


    auto const txbx = surroundingNodes(grow(g1bx, 0, -1), 0);
    auto const tybx = surroundingNodes(grow(g1bx, 1, -1), 1); 
    auto const tzbx = surroundingNodes(grow(g1bx, 2, -1), 2);
    /* Use Minion's Method to do characterstic tracing */
// --------------------X -------------------------------------------------
    AMREX_PARALLEL_FOR_4D(txbx, NCOMP, i, j, k, n, {
        amrex::Real cons1;
        amrex::Real cons2; 
        amrex::Real lo; 
        amrex::Real hi; 
        amrex::Real st;
        amrex::Real fux = (std::abs(umac(i,j,k)) < 1e-10)? 0.e0 : 1.e0; 
        bool uval = umac(i,j,k) >= 0.e0; 
        auto bc = BCs[n];  
        cons1 = (iconserv[n])? - 0.5e0*dt*s(i-1,j,k,n)*divu(i-1,j,k) : 0; 
        cons2 = (iconserv[n])? - 0.5e0*dt*s(i,j,k,n)*divu(i,j,k) : 0; 
        lo    = Ipx(i-1,j,k,n) + 0.5e0*dt*tf(i-1,j,k,n) + cons1; 
        hi    = Imx(i,j,k,n) + 0.5e0*dt*tf(i, j, k, n) + cons2;
        PeleLM_trans_xbc(i, j, k, n, s, lo, hi, umac, bc.lo(0), bc.hi(0), 
                            bx.loVect(), bx.hiVect(), false, false);  
        xlo(i,j,k,n) = lo; 
        xhi(i,j,k,n) = hi;
        st = (uval) ? lo : hi;
        xedge(i, j, k, n) = fux*st + (1.e0 - fux)*0.5e0*(hi + lo);
    }); 

// Clear integral states from PPM 
    Imxeli.clear(); 
    Ipxeli.clear(); 

//--------------------Y -------------------------------------------------
     AMREX_PARALLEL_FOR_4D(tybx, NCOMP, i, j, k, n, {
        amrex::Real cons1;
        amrex::Real cons2; 
        amrex::Real lo; 
        amrex::Real hi; 
        amrex::Real st;
        amrex::Real fuy = (std::abs(vmac(i,j,k)) < 1e-10)? 0.e0 : 1.e0; 

        bool vval = vmac(i,j,k) >= 0.e0; 
        auto bc = BCs[n];  
        cons1 = (iconserv[n])? - 0.5e0*dt*s(i,j-1,k,n)*divu(i,j-1,k) : 0; 
        cons2 = (iconserv[n])? - 0.5e0*dt*s(i,j,k,n)*divu(i,j,k) : 0; 
        lo    = Ipy(i,j-1,k,n) + 0.5e0*dt*tf(i,j-1,k,n) + cons1; 
        hi    = Imy(i,j,k,n)   + 0.5e0*dt*tf(i,j,k,n) + cons2; 
        PeleLM_trans_ybc(i, j, k, n, s, lo, hi, vmac, bc.lo(1), bc.hi(1),
                                 bx.loVect(), bx.hiVect(), false, false);  
        ylo(i,j,k,n) = lo; 
        yhi(i,j,k,n) = hi; 
        st    = (vval >= 0.e0) ? lo : hi; 
        yedge(i, j, k, n) = fuy*st + (1.e0 - fuy)*0.5e0*(hi + lo);
    });

// Clear integral states from PPM 
    Imyeli.clear(); 
    Ipyeli.clear(); 

//------------------ Z ---------------------------------------------------
     AMREX_PARALLEL_FOR_4D(tzbx, NCOMP, i, j, k, n, {
        amrex::Real cons1;
        amrex::Real cons2; 
        amrex::Real lo; 
        amrex::Real hi; 
        amrex::Real st;
        amrex::Real fuz = (std::abs(wmac(i,j,k)) < 1e-10)? 0.e0 : 1.e0; 
        bool wval = wmac(i,j,k) >= 0.e0; 
        auto bc = BCs[n];  
        cons1 = (iconserv[n])? - 0.5e0*dt*s(i,j,k-1,n)*divu(i,j,k-1) : 0; 
        cons2 = (iconserv[n])? - 0.5e0*dt*s(i,j,k,n)*divu(i,j,k) : 0; 
        lo    = Ipz(i,j,k-1,n) + 0.5e0*dt*tf(i,j,k-1,n) + cons1; 
        hi    = Imz(i,j,k,n)   + 0.5e0*dt*tf(i,j,k,n) + cons2; 
        PeleLM_trans_zbc(i, j, k, n, s, lo, hi, wmac, bc.lo(2), bc.hi(2),
                                 bx.loVect(), bx.hiVect(), false, false);  
        zlo(i,j,k,n) = lo; 
        zhi(i,j,k,n) = hi; 
        st    = (wval) ? lo : hi;
        zedge(i, j, k, n) = fuz*st + (1.e0 - fuz)*0.5e0*(hi + lo); 
    });     

// Clear integral states from PPM 
    Imzeli.clear(); 
    Ipzeli.clear(); 
 

//-----------------Create temporary fabs for corner/transverse 

    //X fabs
    FArrayBox xylof(g1bx, NCOMP); 
    FArrayBox xyhif(g1bx, NCOMP); 
    FArrayBox xzlof(g1bx, NCOMP); 
    FArrayBox xzhif(g1bx, NCOMP); 
    Elixir xyle = xylof.elixir(); 
    Elixir xyhe = xyhif.elixir(); 
    Elixir xzle = xzlof.elixir(); 
    Elixir xzhe = xzhif.elixir(); 
    const auto xylo = xylof.array(); 
    const auto xyhi = xyhif.array(); 
    const auto xzlo = xzlof.array(); 
    const auto xzhi = xzhif.array(); 

    //Y fabs
    FArrayBox yxlof(g1bx, NCOMP); 
    FArrayBox yxhif(g1bx, NCOMP); 
    FArrayBox yzlof(g1bx, NCOMP); 
    FArrayBox yzhif(g1bx, NCOMP); 
    Elixir yxle = yxlof.elixir(); 
    Elixir yxhe = yxhif.elixir(); 
    Elixir yzle = yzlof.elixir(); 
    Elixir yzhe = yzhif.elixir(); 
    const auto yxlo = yxlof.array(); 
    const auto yxhi = yxhif.array(); 
    const auto yzlo = yzlof.array(); 
    const auto yzhi = yzhif.array(); 

    //Z fabs 
    FArrayBox zxlof(g1bx, NCOMP); 
    FArrayBox zxhif(g1bx, NCOMP); 
    FArrayBox zylof(g1bx, NCOMP); 
    FArrayBox zyhif(g1bx, NCOMP); 
    Elixir zxle = zxlof.elixir(); 
    Elixir zxhe = zxhif.elixir(); 
    Elixir zyle = zylof.elixir(); 
    Elixir zyhe = zyhif.elixir(); 
    const auto zxlo = zxlof.array(); 
    const auto zxhi = zxhif.array(); 
    const auto zylo = zylof.array(); 
    const auto zyhi = zyhif.array(); 

    auto const xybx = surroundingNodes(grow(bx, 2, 1), 0);
    auto const xzbx = surroundingNodes(grow(bx, 1, 1), 0);  
    auto const yxbx = surroundingNodes(grow(bx, 2, 1), 1); 
    auto const yzbx = surroundingNodes(grow(bx, 0, 1), 1); 
    auto const zxbx = surroundingNodes(grow(bx, 1, 1), 2); 
    auto const zybx = surroundingNodes(grow(bx, 0, 1), 2); 

/*------------------Now perform corner coupling */
    //Dir trans against X
    AMREX_PARALLEL_FOR_4D (yxbx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n]; 
        //YX
        PeleLM_corner_couple(i,j,k, n, dt, dx, iconserv, ylo, yhi, 
                             s, divu, umac, xedge, yxlo, yxhi, 1, 0);
        PeleLM_trans_ybc(i, j, k, n, s, yxlo(i,j,k,n), yxhi(i,j,k,n), vmac,
                                   bc.lo(1), bc.hi(1),
                                   bx.loVect(), bx.hiVect(), true, false);  
    }); 
    
    AMREX_PARALLEL_FOR_4D (zxbx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n]; 
        //ZX
        PeleLM_corner_couple(i,j,k, n, dt, dx, iconserv, zlo, zhi, 
                             s, divu, umac, xedge, zxlo, zxhi, 2, 0);
        PeleLM_trans_zbc(i, j, k, n, s, zxlo(i,j,k,n), zxhi(i,j,k,n), wmac,
                                   bc.lo(2), bc.hi(2),
                                   bx.hiVect(), bx.loVect(), true, false); 
    });

   //Dir trans against Y
    AMREX_PARALLEL_FOR_4D (xybx, NCOMP, i, j, k, n, {
        //XY
        const auto bc = BCs[n]; 
        PeleLM_corner_couple(i,j,k, n, dt, dy, iconserv, xlo, xhi, 
                             s, divu, vmac, yedge, xylo, xyhi, 0, 0);
        PeleLM_trans_xbc(i, j, k, n, s, xylo(i,j,k,n), xyhi(i,j,k,n), umac,
                            bc.lo(0), bc.hi(0), 
                            bx.loVect(), bx.hiVect(), true, false);
    }); 

    AMREX_PARALLEL_FOR_4D (zybx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n]; 
        //ZY 
        PeleLM_corner_couple(i,j,k, n, dt, dy, iconserv, zlo, zhi, 
                             s, divu, vmac, yedge, zylo, zyhi, 2, 1);
        PeleLM_trans_zbc(i, j, k, n, s, zylo(i,j,k,n), zyhi(i,j,k,n), wmac,
                                 bc.lo(2), bc.hi(2),
                                 bx.hiVect(), bx.loVect(), false, true); 
    });
    //Dir trans against Z
    AMREX_PARALLEL_FOR_4D (xzbx, NCOMP, i, j, k, n, {
        //XZ
        const auto bc = BCs[n]; 
        PeleLM_corner_couple(i,j,k, n, dt, dz, iconserv, xlo, xhi, 
                             s, divu, wmac, zedge, xzlo, xzhi, 0, 1);
        PeleLM_trans_xbc(i, j, k, n, s, xzlo(i,j,k,n), xzhi(i,j,k,n), umac,
                            bc.lo(0), bc.hi(0), 
                            bx.loVect(), bx.hiVect(), false, true); 
    });  

    AMREX_PARALLEL_FOR_4D (yzbx, NCOMP, i, j, k, n, {
        const auto bc = BCs[n]; 
        //YZ
        PeleLM_corner_couple(i,j,k, n, dt, dz, iconserv, ylo, yhi, 
                             s, divu, wmac, zedge, yzlo, yzhi, 1, 1);
        PeleLM_trans_ybc(i, j, k, n, s, yzlo(i,j,k,n), yzhi(i,j,k,n), vmac,
                                   bc.lo(1), bc.hi(1),
                                   bx.loVect(), bx.hiVect(), false, true);
    });

   /* Upwinding Edge States */ 
   const auto xbx = surroundingNodes(bx, 0); 
   const auto ybx = surroundingNodes(bx, 1); 
   const auto zbx = surroundingNodes(bx, 2);
 
   AMREX_PARALLEL_FOR_4D (xybx, NCOMP,  i, j, k, n, { 
        amrex::Real fu = (std::abs(umac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        amrex::Real st; 
        st = (umac(i,j,k) >= 0)? xylo(i,j,k,n) : xyhi(i,j,k,n); 
        xylo(i,j,k,n) = fu*st 
                      + (1. - fu)*0.5*(xyhi(i,j,k,n) + xylo(i,j,k,n)); //This is safe
    }); 
    
   AMREX_PARALLEL_FOR_4D (xybx, NCOMP,  i, j, k, n, { 
        amrex::Real fu = (std::abs(umac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        amrex::Real st; 
        st = (umac(i,j,k) >= 0)? xzlo(i,j,k,n) : xzhi(i,j,k,n); 
        xzlo(i,j,k,n) = fu*st 
                      + (1. - fu)*0.5*(xzhi(i,j,k,n) + xzlo(i,j,k,n)); 
    });
 
   AMREX_PARALLEL_FOR_4D (yxbx, NCOMP, i, j, k, n, { 
        amrex::Real fu = (std::abs(vmac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        amrex::Real st; 
        st = (vmac(i,j,k) >= 0)? yxlo(i,j,k,n) : yxhi(i,j,k,n); 
        yxlo(i,j,k,n) = fu*st 
                      + (1. - fu)*0.5*(yxhi(i,j,k,n) + yxlo(i,j,k,n));
    }); 

   AMREX_PARALLEL_FOR_4D (yzbx, NCOMP, i, j, k, n, { 
        amrex::Real fu = (std::abs(vmac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        amrex::Real st; 
        st = (umac(i,j,k) >= 0)? yzlo(i,j,k,n) : yzhi(i,j,k,n); 
        yzlo(i,j,k,n) = fu*st 
                      + (1. - fu)*0.5*(yzhi(i,j,k,n) + yzlo(i,j,k,n)); 
    });
 
   AMREX_PARALLEL_FOR_4D (zxbx, NCOMP, i, j, k, n,{ 
        amrex::Real fu = (std::abs(wmac(i,j,k)) < 1e-10)? 0.0 : 1.0;  
        amrex::Real st; 
        st = (wmac(i,j,k) >= 0)? zxlo(i,j,k,n) : zxhi(i,j,k,n); 
        zxlo(i,j,k,n) = fu*st 
                      + (1. - fu)*0.5*(zxhi(i,j,k,n) + zxlo(i,j,k,n));
    }); 

   AMREX_PARALLEL_FOR_4D (zybx, NCOMP, i, j, k, n,{ 
        amrex::Real fu = (std::abs(wmac(i,j,k)) < 1e-10)? 0.0 : 1.0;  
        amrex::Real st; 
        st = (wmac(i,j,k) >= 0)? zylo(i,j,k,n) : zyhi(i,j,k,n); 
        zylo(i,j,k,n) = fu*st 
                      + (1. - fu)*0.5*(zyhi(i,j,k,n) + zylo(i,j,k,n)); 
    }); 

    /* Final Update of Faces */ 

    AMREX_PARALLEL_FOR_4D(xbx, NCOMP, i, j, k, n, { 
        amrex::Real stl;
        amrex::Real sth;
        amrex::Real temp; 
        auto bc = BCs[n]; 
//--------------------------------------- X -------------------------------------- 
        if(iconserv[n]){
        stl = xlo(i,j,k,n) - (0.5*dt/dy)*(yzlo(i-1,j+1,k,n)*vmac(i-1,j+1,k)
                           - yzlo(i-1,j,k,n)*vmac(i-1,j,k))
                           - (0.5*dt/dz)*(zylo(i-1,j,k+1,n)*wmac(i-1,j,k+1)
                           - zylo(i-1,j,k,n)*wmac(i-1,j,k))
                           + (0.5*dt/dy)*s(i-1,j,k,n)*(vmac(i-1,j+1,k) -vmac(i-1,j,k))
                           + (0.5*dt/dz)*s(i-1,j,k,n)*(wmac(i-1,j,k+1) -wmac(i-1,j,k));

        sth = xhi(i,j,k,n) - (0.5*dt/dy)*(yzlo(i,j+1,k,n)*vmac(i,j+1,k)
                           - yzlo(i,j,k,n)*vmac(i,j,k))
                           - (0.5*dt/dz)*(zylo(i,j,k+1,n)*wmac(i,j,k+1)
                           - zylo(i,j,k,n)*wmac(i,j,k))
                           + (0.5*dt/dy)*s(i,j,k,n)*(vmac(i,j+1,k) -vmac(i,j,k))
                           + (0.5*dt/dz)*s(i,j,k,n)*(wmac(i,j,k+1) -wmac(i,j,k)); 
        }
        else{
        stl = xlo(i,j,k,n) - (0.25*dt/dy)*(vmac(i-1,j+1,k)+vmac(i-1,j,k))*
                             (yzlo(i-1,j+1,k,n) - yzlo(i-1,j,k,n))
                           - (0.25*dt/dz)*(wmac(i-1,j,k+1)+wmac(i-1,j,k))*
                             (zylo(i-1,j,k+1,n) - zylo(i-1,j,k,n));

        sth = xhi(i,j,k,n) - (0.25*dt/dy)*(vmac(i,j+1,k)+vmac(i,j,k))*
                             (yzlo(i,j+1,k,n) - yzlo(i,j,k,n))
                           - (0.25*dt/dz)*(wmac(i,j,k+1)+wmac(i,j,k))*
                             (zylo(i,j,k+1,n) - zylo(i,j,k,n));
        }
        PeleLM_trans_xbc(i, j, k, n, s, stl, sth, umac, bc.lo(0), bc.hi(0),
                                 bx.loVect(), bx.hiVect(), true, true);  
        temp = (umac(i,j,k) >= 0.e0) ? stl : sth; 
        temp = (std::abs(umac(i,j,k)) < 1e-10) ? 0.5*(stl + sth) : temp; 
        xstate(i,j,k,n) = temp;
     }); 

    AMREX_PARALLEL_FOR_4D(ybx, NCOMP, i, j, k, n, { 
        amrex::Real stl;
        amrex::Real sth;
        amrex::Real temp; 
        auto bc = BCs[n]; 
//-------------------------------------- Y ------------------------------------            
        if(iconserv[n]){
        stl = ylo(i,j,k,n) - (0.5*dt/dx)*(xzlo(i+1,j-1,k,n)*umac(i+1,j-1,k)
                           - xzlo(i,j-1,k,n)*umac(i,j-1,k))
                           - (0.5*dt/dz)*(zxlo(i,j-1,k+1,n)*wmac(i,j-1,k+1)
                           - zxlo(i,j-1,k,n)*wmac(i,j-1,k))
                           + (0.5*dt/dx)*s(i,j-1,k,n)*(umac(i+1,j-1,k) -umac(i,j-1,k))
                           + (0.5*dt/dz)*s(i,j-1,k,n)*(wmac(i,j-1,k+1) -wmac(i,j-1,k));

        sth = yhi(i,j,k,n) - (0.5*dt/dx)*(xzlo(i+1,j,k,n)*umac(i+1,j,k)
                           - xzlo(i,j,k,n)*umac(i,j,k))
                           - (0.5*dt/dz)*(zxlo(i,j,k+1,n)*wmac(i,j,k+1)
                           - zxlo(i,j,k,n)*wmac(i,j,k))
                           + (0.5*dt/dx)*s(i,j,k,n)*(umac(i+1,j,k) -umac(i,j,k))
                           + (0.5*dt/dz)*s(i,j,k,n)*(wmac(i,j,k+1) -wmac(i,j,k)); 
        }
        else{
        stl = ylo(i,j,k,n) - (0.25*dt/dx)*(umac(i+1,j-1,k)+umac(i,j-1,k))*
                             (xzlo(i+1,j-1,k,n) - xzlo(i,j-1,k,n))
                           - (0.25*dt/dz)*(wmac(i,j-1,k+1)+wmac(i,j-1,k))*
                             (zxlo(i,j-1,k+1,n) - zxlo(i,j-1,k,n));

        sth = yhi(i,j,k,n) - (0.25*dt/dx)*(umac(i+1,j,k)+umac(i,j,k))*
                             (xzlo(i+1,j,k,n) - xzlo(i,j,k,n))
                           - (0.25*dt/dz)*(wmac(i,j,k+1)+wmac(i,j,k))*
                             (zxlo(i,j,k+1,n) - zxlo(i,j,k,n));
        }
        PeleLM_trans_ybc(i, j, k, n, s, stl, sth, vmac, bc.lo(1), bc.hi(1), 
                                  bx.loVect(), bx.hiVect(), true, true); 

        temp = (vmac(i,j,k) >= 0.e0) ? stl : sth; 
        temp = (std::abs(vmac(i,j,k)) < 1e-10) ? 0.5*(stl + sth) : temp; 
        ystate(i,j,k,n) = temp;
      });

     AMREX_PARALLEL_FOR_4D(zbx, NCOMP, i, j, k, n, { 
        amrex::Real stl;
        amrex::Real sth;
        amrex::Real temp; 
        auto bc = BCs[n]; 
//----------------------------------- Z ----------------------------------------- 
        if(iconserv[n]){
        stl = zlo(i,j,k,n) - (0.5*dt/dx)*(xylo(i+1,j,k-1,n)*umac(i+1,j,k-1)
                           - xylo(i,j,k-1,n)*umac(i,j,k-1))
                           - (0.5*dt/dy)*(yxlo(i,j+1,k-1,n)*vmac(i,j+1,k-1)
                           - yxlo(i,j,k-1,n)*vmac(i,j,k-1))
                           + (0.5*dt/dx)*s(i,j,k-1,n)*(umac(i+1,j,k-1) -umac(i,j,k-1))
                           + (0.5*dt/dy)*s(i,j,k-1,n)*(vmac(i,j+1,k-1) -vmac(i,j,k-1));

        sth = zhi(i,j,k,n) - (0.5*dt/dx)*(xylo(i+1,j,k,n)*umac(i+1,j,k)
                           - xylo(i,j,k,n)*umac(i,j,k))
                           - (0.5*dt/dy)*(yxlo(i,j+1,k,n)*vmac(i,j+1,k)
                           - yxlo(i,j,k,n)*vmac(i,j,k))
                           + (0.5*dt/dx)*s(i,j,k,n)*(umac(i+1,j,k) -umac(i,j,k))
                           + (0.5*dt/dy)*s(i,j,k,n)*(vmac(i,j+1,k) -vmac(i,j,k)); 
        }
        else{
        stl = zlo(i,j,k,n) - (0.25*dt/dx)*(umac(i+1,j,k-1)+umac(i,j,k-1))*
                             (xylo(i+1,j,k-1,n) - xylo(i,j,k-1,n))
                           - (0.25*dt/dy)*(vmac(i,j+1,k-1)+vmac(i,j,k-1))*
                             (yxlo(i,j+1,k-1,n) - yxlo(i,j,k-1,n));

        sth = zhi(i,j,k,n) - (0.25*dt/dx)*(umac(i+1,j,k)+umac(i,j,k))*
                             (xylo(i+1,j,k,n) - xylo(i,j,k,n))
                           - (0.25*dt/dy)*(vmac(i,j+1,k)+vmac(i,j,k))*
                             (yxlo(i,j+1,k,n) - yxlo(i,j,k,n));
        }
        PeleLM_trans_zbc(i, j, k, n, s, stl, sth, wmac, bc.lo(2), bc.hi(2), 
                                        bx.loVect(), bx.hiVect(), true, true);  
        temp = (wmac(i,j,k) >= 0.e0) ? stl : sth; 
        temp = (std::abs(wmac(i,j,k)) < 1e-10) ? 0.5*(stl + sth) : temp; 
        zstate(i,j,k,n) = temp;
    }); 
}
