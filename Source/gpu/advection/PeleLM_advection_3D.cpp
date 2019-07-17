#include "PeleLM_advection_3D.H" 

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
                       const int iconserv[]) 
{    
    auto const g1bx = amrex::grow(bx, 1); 
    const amrex::Real dx = del[0]; 
    const amrex::Real dy = del[1]; 
    const amrex::Real dz = del[2]; 
    
    auto const gxbx = amrex::grow(g1bx,0, 1); 
    FArrayBox Imxf(gxbx, NVAR); 
    FArrayBox Ipxf(gxbx, NVAR);
    Elixir Imxeli = Imxf.elixir(); 
    Elixir Ipxeli = Ipxf.elixir(); 
    auto const Imx = Imxf.array(); 
    auto const Ipx = Ipxf.array(); 

    auto const gybx = amrex::grow(g1bx,1, 1); 
    FArrayBox Imyf(gybx, NVAR); 
    FArrayBox Ipyf(gybx, NVAR);
    Elixir Imyeli = Imyf.elixir(); 
    Elixir Ipyeli = Ipyf.elixir(); 
    auto const Imy = Imyf.array(); 
    auto const Ipy = Ipyf.array(); 

   
    auto const gzbx = amrex::grow(g1bx,2, 1); 
    FArrayBox Imzf(gzbx, NVAR); 
    FArrayBox Ipzf(gzbx, NVAR);
    Elixir Imzeli = Imzf.elixir(); 
    Elixir Ipzeli = Ipzf.elixir(); 
    auto const Imz = Imzf.array(); 
    auto const Ipz = Ipzf.array(); 

    /* Use PPM to generate Im and Ip */ 
    AMREX_PARALLEL_FOR_3D(g1bx, i, j, k, {
        for(int n = 0; n < NVAR; ++n){
            PeleLM_ppm(i, j, k, n, dt, dx, s, umac, Imx, Ipx, 0); 
            PeleLM_ppm(i, j, k, n, dt, dy, s, vmac, Imy, Ipy, 1); 
            PeleLM_ppm(i, j, k, n, dt, dz, s, wmac, Imz, Ipz, 2); 
        }
    }); 


    /* TODO do BC's */ 
    auto const xbx = amrex::grow(bx,0, 1); 
    FArrayBox xlf(xbx, NVAR); 
    FArrayBox xhf(xbx, NVAR);
    Elixir xleli = xlf.elixir(); 
    Elixir xheli = xhf.elixir(); 
    auto const xl = xlf.array(); 
    auto const xh = xhf.array(); 

    auto const ybx = amrex::grow(bx,1, 1); 
    FArrayBox ylf(ybx, NVAR); 
    FArrayBox yhf(ybx, NVAR);
    Elixir yleli = ylf.elixir(); 
    Elixir yheli = yhf.elixir(); 
    auto const yl = ylf.array(); 
    auto const yh = yhf.array(); 

   
    auto const zbx = amrex::grow(bx,2, 1); 
    FArrayBox zlf(zbx, NVAR); 
    FArrayBox zhf(zbx, NVAR);
    Elixir zleli = zlf.elixir(); 
    Elixir zheli = zhf.elixir(); 
    auto const zl = zlf.array(); 
    auto const zh = zhf.array();

    /* Use Minion's Method to do characterstic tracing */
    AMREX_PARALLEL_FOR_3D(bx, i, j, k, {
        amrex::Real cons1, cons2, lo, hi, st;
        amrex::Real fu = (std::abs(umac(i, j, k)) < 1e-10) ? 0.e0 : 1.e0; 
        bool uval = umac(i,j,k) >= 0.e0; 
        bool vval = vmac(i,j,k) >= 0.e0; 
        bool wval = wmac(i,j,k) >= 0.e0; 
        for(int n = 0; n < NVAR; ++n){ 
            cons1 = (iconserv[n])? - 0.5e0*dt*s(i-1,j,k,n)*divu(i-1,j,k) : 0; 
            cons2 = (iconserv[n])? - 0.5e0*dt*s(i,j,k,n)*divu(i,j,k) : 0; 
            lo    = Ipx(i-1,j,k,n) + 0.5e0*dt*tf(i-1,j,k,n) + cons1; 
            hi    = Imx(i,j,k,n) + 0.5e0*dt*tf(i, j, k, n) + cons2;
            xlo(i,j,k,n) = lo; 
            xhi(i,j,k,n) = hi; 
            st = (uval) ? lo : hi; 
            xstate(i, j, k, n) = fu*st + (1.e0 - fu)*0.5e0*(hi + lo); 
        }

        fu = (std::abs(vmac(i,j,k,)) < 1e-10)? 0.e0 : 1.e0; 
        for(int n = 0; n <NVAR; ++n){
            cons1 = (iconserv[n])? - 0.5e0*dt*s(i,j-1,k,n)*divu(i,j-1,k) : 0; 
            cons2 = (iconserv[n])? - 0.5e0*dt*s(i,j,k,n)*divu(i,j,k) : 0; 
            lo    = Ipy(i,j-1,k,n) + 0.5e0*dt*tf(i,j-1,k,n) + cons1; 
            hi    = Imy(i,j,k,n)   + 0.5e0*dt*tf(i,j,k,n) + cons2; 
            ylo(i,j,k,n) = lo; 
            yhi(i,j,k,n) = hi; 
            st    = (vval >= 0.e0) ? lo : hi; 
            ystate(i, j, k, n) = fu*st + (1.e0 - fu)*0.5e0*(hi + lo); 
        }

        fu = (std::abs(wmac(i,j,k,)) < 1e-10)? 0.e0 : 1.e0; 
        for(int n = 0; n <NVAR; ++n){
            cons1 = (iconserv[n])? - 0.5e0*dt*s(i,j,k-1,n)*divu(i,j,k-1) : 0; 
            cons2 = (iconserv[n])? - 0.5e0*dt*s(i,j,k,n)*divu(i,j,k) : 0; 
            lo    = Ipz(i,j,k-1,n) + 0.5e0*dt*tf(i,j,k-1,n) + cons1; 
            hi    = Imz(i,j,k,n)   + 0.5e0*dt*tf(i,j,k,n) + cons2; 
            zlo(i,j,k,n) = lo; 
            zhi(i,j,k,n) = hi; 
            st    = (wval) ? lo : hi;
            zstate(i, j, k, n) = fu*st + (1.e0 - fu)*0.5e0*(hi + lo); 
        }
    });     


// Clear integral states from PPM 
    Imxeli.clear(); 
    Ipxeli.clear(); 
    Imyeli.clear(); 
    Ipyeli.clear(); 
    Imzeli.clear(); 
    Ipzeli.clear(); 
 

//-----------------Create temporary fabs for corner/transverse 

    //X fabs
    FArrayBox xylof(bx, NVAR); 
    FArrayBox xyhif(bx, NVAR); 
    FArrayBox xzlof(bx, NVAR); 
    FArrayBox xzhif(bx, NVAR); 
    Elixir xyle = xylof.elixir(); 
    Elixir xyhe = xyhif.elixir(); 
    Elixir xzle = xzlof.elixir(); 
    Elixir xzhe = xzhif.elixir(); 
    const auto xylo = xylof.array(); 
    const auto xyhi = xyhif.array(); 
    const auto xzlo = xzlof.array(); 
    const auto xzhi = xzhif.array(); 

    //Y fabs
    FArrayBox yxlof(bx, NVAR); 
    FArrayBox yxhif(bx, NVAR); 
    FArrayBox yzlof(bx, NVAR); 
    FArrayBox yzhif(bx, NVAR); 
    Elixir xyle = yxlof.elixir(); 
    Elixir xyhe = yxhif.elixir(); 
    Elixir xzle = zylof.elixir(); 
    Elixir xzhe = zyhif.elixir(); 
    const auto yxlo = yxlof.array(); 
    const auto yxhi = yxhif.array(); 
    const auto yzlo = yzlof.array(); 
    const auto yzhi = yzhif.array(); 

    //Z fabs 
    FArrayBox zxlof(bx, NVAR); 
    FArrayBox zxhif(bx, NVAR); 
    FArrayBox zylof(bx, NVAR); 
    FArrayBox zyhif(bx, NVAR); 
    Elixir zxle = zxlof.elixir(); 
    Elixir zxhe = zxhif.elixir(); 
    Elixir zyle = zylof.elixir(); 
    Elixir zyhe = zyhif.elixir(); 
    const auto zxlo = zxlof.array(); 
    const auto zxhi = zxhif.array(); 
    const auto zylo = zylof.array(); 
    const auto zyhi = zyhif.array(); 

/*------------------Now perform corner coupling */ 
    AMREX_PARALLEL_FOR_3D(bx, i, j, k, {
        for(int n = 0; n < NVAR; ++n){
        //X trans 
        PeleLM_corner_couple(i,j,k, n, dt, dy, dz, iconserv, xlo, xhi, 
                             s, divu, vmac, wmac, ystate, zstate, xylo, xyhi, 
                             xzlo, xzhi, 0); 
        //Y trans 
        PeleLM_corner_couple(i,j,k, n, dt, dx, dz, iconserv, ylo, yhi, 
                             s, divu, umac, wmac, xstate, zstate, yxlo, yxhi, 
                             yzlo, yzhi, 1); 

        //Z trans 
        PeleLM_corner_couple(i,j,k, n, dt, dx, dy, iconserv, zlo, zhi, 
                             s, divu, umac, vmac, xstate, ystate, zxlo, zxhi, 
                             zylo, zyhi, 2); 

        }
    });

    /*TODO Trans_bc */ 

   /* Upwinding Edge States */ 
   AMREX_PARALEL_FOR_3D(bx, i, j, k, { 
        amrex::Real fu = (std::abs(umac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        amrex::Real st; 
        for(int n = 0; n < NVAR; ++n){
            st = (umac(i,j,k) >= 0)? xylo(i,j,k,n) : xyhi(i,j,k,n); 
            xylo(i,j,k,n) = fu*st 
                          + (1. - fu)*0.5*(xyhi(i,j,k,n) + xylo(i,j,k,n)); //This is safe

            st = (umac(i,j,k) >= 0)? xzlo(i,j,k,n) : xzhi(i,j,k,n); 
            xzlo(i,j,k,n) = fu*st 
                          + (1. - fu)*0.5*(xzhi(i,j,k,n) + xzlo(i,j,k,n)); 
        }

        fu = (std::abs(vmac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        for(int n = 0; n < NVAR; ++n){
            st = (vmac(i,j,k) >= 0)? yxlo(i,j,k,n) : yxhi(i,j,k,n); 
            yxlo(i,j,k,n) = fu*st 
                          + (1. - fu)*0.5*(yxhi(i,j,k,n) + yxlo(i,j,k,n));

            st = (umac(i,j,k) >= 0)? yzlo(i,j,k,n) : yzhi(i,j,k,n); 
            yzlo(i,j,k,n) = fu*st 
                          + (1. - fu)*0.5*(yzhi(i,j,k,n) + yzlo(i,j,k,n)); 
        }

        fu = (std;:abs(wmac(i,j,k)) < 1e-10)? 0.0 : 1.0; 
        for(int n = 0; n < NVAR; ++n){
            st = (wmac(i,j,k) >= 0)? zxlo(i,j,k,n) : zxhi(i,j,k,n); 
            zxlo(i,j,k,n) = fu*st 
                          + (1. - fu)*0.5*(zxhi(i,j,k,n) + zxlo(i,j,k,n));

            st = (wmac(i,j,k) >= 0)? zylo(i,j,k,n) : zyhi(i,j,k,n); 
            zylo(i,j,k,n) = fu*st 
                          + (1. - fu)*0.5*(zyhi(i,j,k,n) + zylo(i,j,k,n)); 
        }
    }); 

    /* Final Update of Faces */ 

    AMREX_PARALLEL_FOR_3D(bx, i, j, k, { 
        amrex::Real stl, sth, temp; 
        for(int n = 0; n < NVAR: ++n){
//--------------------------------------- X -------------------------------------- 
            if(iconserv(n)){
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
            //TODO BC handling 
            temp = (umac(i,j,k) >= 0.e0) ? stl : sth; 
            temp = (std::abs(umac(i,j,k)) < 1e-10) ? 0.5*(stl + sth) : temp; 
            xstate(i,j,k,n) = temp; 
//-------------------------------------- Y ------------------------------------            
            if(iconserv(n)){
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
            stl = ylo(i,j,k,n) - (0.25*dt/dx)*(vmac(i+1,j-1,k)+vmac(i,j-1,k))*
                                 (yzlo(i+1,j-1,k,n) - yzlo(i,j-1,k,n))
                               - (0.25*dt/dz)*(wmac(i,j-1,k+1)+wmac(i,j-1,k))*
                                 (zylo(i,j-1,k+1,n) - zylo(i,j-1,k,n));
 
            sth = yhi(i,j,k,n) - (0.25*dt/dx)*(vmac(i+1,j,k)+vmac(i,j,k))*
                                 (yzlo(i+1,j,k,n) - yzlo(i,j,k,n))
                               - (0.25*dt/dz)*(wmac(i,j,k+1)+wmac(i,j,k))*
                                 (zylo(i,j,k+1,n) - zylo(i,j,k,n));
            }
            //TODO BC handling 
            temp = (vmac(i,j,k) >= 0.e0) ? stl : sth; 
            temp = (std::abs(vmac(i,j,k)) < 1e-10) ? 0.5*(stl + sth) : temp; 
            ystate(i,j,k,n) = temp; 
//----------------------------------- Z ----------------------------------------- 
            if(iconserv(n)){
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
            //TODO BC handling 
            temp = (wmac(i,j,k) >= 0.e0) ? stl : sth; 
            temp = (std::abs(wmac(i,j,k)) < 1e-10) ? 0.5*(stl + sth) : temp; 
            zstate(i,j,k,n) = temp; 
        }
    }); 



}
