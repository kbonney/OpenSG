from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from dolfinx.fem import form, petsc, Function, locate_dofs_topological
from ufl import TrialFunction, TestFunction, rhs, as_tensor, dot, SpatialCoordinate, Measure
import petsc4py.PETSc
from dolfinx.fem.petsc import assemble_matrix

from mpi4py import MPI
import ufl
from contextlib import ExitStack


def generate_boundary_markers(xmin, xmax):
    def is_left_boundary(x):
        return np.isclose(x[0], xmin,atol=0.01)
    def is_right_boundary(x):
        return np.isclose(x[0], xmax,atol=0.01)
    return is_left_boundary, is_right_boundary

def Rsig(frame):   # ROTATION MATRIX IN UFL Form 

        b11,b12,b13=frame[0][0],frame[1][0],frame[2][0]
        b21,b22,b23=frame[0][1],frame[1][1],frame[2][1]
        b31,b32,b33=frame[0][2],frame[1][2],frame[2][2]
        
        return as_tensor([(b11*b11, b12*b12, b13*b13, 2*b12*b13, 2*b11*b13,2* b11*b12),
                     (b21*b21, b22*b22, b23*b23, 2*b22*b23, 2*b21*b23, 2*b21*b22),
                     (b31*b31, b32*b32, b33*b33, 2*b32*b33, 2*b31*b33, 2*b31*b32),
                     (b21*b31, b22*b32, b23*b33, b23*b32+b22*b33, b23*b31+b21*b33, b22*b31+b21*b32),
                     (b11*b31, b12*b32, b13*b33, b13*b32+b12*b33, b13*b31+b11*b33, b12*b31+b11*b32),
                     (b11*b21, b12*b22, b13*b23, b13*b22+b12*b23, b13*b21+b11*b23, b12*b21+b11*b22)])

def C(i,frame,material_parameters):  # Stiffness matrix
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12 
    CC=as_tensor(np.linalg.inv(S))
    R_sig=Rsig(frame)
    return dot(dot(R_sig,CC),R_sig.T) 

def mass_boun(x,dx,density,nphases): # Mass matrix
    mu= assemble_scalar(form(sum([density[i]*dx(i) for i in range(nphases)])))
    xm2=(1/mu)*assemble_scalar(form(sum([x[1]*density[i]*dx(i) for i in range(nphases)])))
    xm3=(1/mu)*assemble_scalar(form(sum([x[2]*density[i]*dx(i) for i in range(nphases)])))
    i22=assemble_scalar(form(sum([(x[2]**2)*density[i]*dx(i) for i in range(nphases)])))
    i33=assemble_scalar(form(sum([(x[1]**2)*density[i]*dx(i) for i in range(nphases)])))    
    i23=assemble_scalar(form(sum([x[1]*x[2]*density[i]*dx(i) for i in range(nphases)])))
    return np.array([(mu,0,0,0,mu*xm3,-mu*xm2),
                      (0,mu,0,-mu*xm3,0,0),
                      (0,0,mu,mu*xm2,0,0),
                      (0,-mu*xm3, mu*xm2, i22+i33, 0,0),
                      (mu*xm3, 0,0,0,i22,i23),
                      (-mu*xm2,0,0,0,i23,i33)])  
    
def solve_ksp(A, F, V):
    """Krylov Subspace Solver for Aw = F

    Parameters
    ----------
    A : array
        stiffness matrix
    F : array
        Load or force vector
    V : function space
        _description_

    Returns
    -------
    array
        solution vector (displacement field)
    """
    w = Function(V)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm = MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl = 24, ival = 1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl = 25, ival = 0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F, w.vector)
    w.vector.ghostUpdate(
        addv = petsc4py.PETSc.InsertMode.INSERT, mode = petsc4py.PETSc.ScatterMode.FORWARD
    )
    ksp.destroy()
    return w.vector[:]

def compute_nullspace(V):
    """Compute nullspace to restrict Rigid body motions

    Constructs a translational null space for the vector-valued function space V
    and ensures that it is properly orthonormalized.

    Parameters
    ----------
    V : functionspace
        _description_

    Returns
    -------
    NullSpace
        Nullspace of V
    """
    # extract the Index Map from the Function Space
    index_map = V.dofmap.index_map

    # initialize nullspace basis with petsc vectors
    nullspace_basis = [
        dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs)
        for _ in range(4)
    ]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]

    # identify the degrees of freedom indices for each subspace (x, y, z)
    dofs = [V.sub(i).dofmap.list for i in range(3)]

    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0
        
    xx = V.tabulate_dof_coordinates()
    xx = xx.reshape((-1, 3))
        
    for i in range(len(xx)):  # Build twist nullspace
            basis[3][3*i+1]=-xx[i,2]
            basis[3][3*i+2]=xx[i,1] 
    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)
    
    ret_val = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm = MPI.COMM_WORLD)

    return ret_val

def gamma_e(x):
    """_summary_

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    gamma_e = as_tensor([
        (1, 0, x[2], -x[1]),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, x[1], 0, 0),
        (0, -x[2], 1, 0),
    ])
    
    return gamma_e


def Dee_(x, C):
        x2,x3=x[1],x[2]
        return as_tensor([
            (C[0,0],               C[0,4]*x2-C[0,5]*x3,                                 C[0,0]*x3,                -C[0,0]*x2),
            (C[4,0]*x2-Cc[5,0]*x3, x2*(C[4,4]*x2-C[5,4]*x3)-x3*(C[4,5]*x2-C[5,5]*x3), x3*(C[4,0]*x2-Cc[5,0]*x3), -x2*(C[4,0]*x2-C[5,0]*x3)),
            (C[0,0]*x3,            x3*(C[0,4]*x2-C[0,5]*x3),                            C[0,0]*x3**2,              -C[0,0]*x2*x3),
            (-C[0,0]*x2,           -x2*(C[0,4]*x2-C[0,5]*x3),                           -C[0,0]*x2*x3,              C[0,0]*x2**2)])
            

def gamma_h(dx,v,dim):    
    aa,b=1,2
    if dim==2:
        ret_val=as_vector([0,v[1].dx(aa),v[2].dx(b),v[1].dx(b)+v[2].dx(aa),v[0].dx(b),v[0].dx(aa)])
    elif dim==3:
        ret_val=as_vector([v[0].dx(0),v[1].dx(aa),v[2].dx(b),v[1].dx(b)+v[2].dx(aa),v[0].dx(b)+v[b].dx(0),v[0].dx(aa)+v[aa].dx(0)])
    return ret_val

def gamma_l(e,x,w): 
    # e,x required as element can be of left/right boundary or quad mesh
    ret_val = as_vector([v[0],0,0,0, v[2],v[1]]) 
    return ret_val


def local_boun(mesh, frame, subdomains):

    V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "CG", mesh.topology.cell_name(), 1, shape = (3, )))
    le1, le2, le3 = frame
    e1l, e2l, e3l = Function(V), Function(V), Function(V)
    
    fexpr1 = dolfinx.fem.Expression(le1,V.element.interpolation_points(), comm = MPI.COMM_WORLD)
    e1l.interpolate(fexpr1) 
    
    fexpr2 = dolfinx.fem.Expression(le2,V.element.interpolation_points(), comm = MPI.COMM_WORLD)
    e2l.interpolate(fexpr2) 
    
    fexpr3 = dolfinx.fem.Expression(le3,V.element.interpolation_points(), comm = MPI.COMM_WORLD)
    e3l.interpolate(fexpr3) 
    
    frame = [e1l,e2l,e3l]
    dv = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    dx = Measure('dx')(domain=mesh, subdomain_data=subdomains)
        
    return frame, V, dv, v, x, dx
          
def A_mat(dx_l,v_l,dvl,dc_matrix_l, nphases,nullspace):
    """Assembly matrix

    Parameters
    ----------
    ABD : _type_
        _description_
    e_l : _type_
        _description_
    x_l : _type_
        _description_
    dx_l : _type_
        _description_
    nullspace_l : _type_
        _description_
    v_l : _type_
        _description_
    dvl : _type_
        _description_
    nphases : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    F2 = sum([dot(dot(C(i,dc_matrix_l),gamma_h(dx_l,dvl)),gamma_h(dx_l,v_l))*dx_l(i) for i in range(nphases)])  
    A_l = assemble_matrix(form(F2))
    A_l.assemble()
    A_l.setNullSpace(nullspace_l) 
    return A_l

def timo_boun(meshdata,material_parameters):
    nphases=len(material_parameters)

    mesh = meshdata["mesh"]
    frame = meshdata["frame"]
    subdomains = meshdata["subdomains"]
    nullspace = meshdata["nullspace"]
    
    e, V, dv, v_, x, dx = local_boun(mesh,frame,subdomains)          
    mesh.topology.create_connectivity(2, 2)
    A = A_mat(ABD, e,x,dx,compute_nullspace(V),v_,dv, nphases,nullspace)
    V0,Dle,Dhe,Dee,V1s=initialize_array(V)
    
    for p in range(4):
        F2=sum([dot(dot(C(i,frame),gamma_e(x)[:,p]),gamma_h(dx,v_))*dx(i) for i in range(nphases)])  
        r_he = form(rhs(F2))
        F = petsc.assemble_vector(r_he)
        F.ghostUpdate(addv = petsc4py.PETSc.InsertMode.ADD, mode = petsc4py.PETSc.ScatterMode.REVERSE)
        nullspace.remove(F)
        Dhe[:,p]=F[:]
        V0[:,p] = solve_ksp(A,F,V)
        
    V0_csr=csr_matrix(V0) 
    D1=V0_csr.T.dot(csr_matrix(-Dhe)) # C(i,frame)

    for s in range(4):
        for k in range(4): 
            f=dolfinx.fem.form(sum([Dee_(x,C(i,frame))[s,k]*dx(i) for i in range(nphases)]))
            Dee[s,k]=dolfinx.fem.assemble_scalar(f)
            
    D_eff= Dee + D1 # Effective Stiffness Matrix (EB)

    # Boundary timoshenko matrix
    F1=sum([dot(dot(C(i,frame),gamma_l(dv)),gamma_l(v_))*dx(i) for i in range(nphases)])
    a1=form(F1)
    Dll=assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av=Dll.getValuesCSR()
    Dll=csr_matrix((av, aj, ai)) 

    for p in range(4):
            F1=sum([dot(dot(C(i,frame),gamma_e(x)[:,p]),gamma_l(v_))*dx(i) for i in range(nphases)])
            Dle[:,p]= petsc.assemble_vector(form(F1))[:] 
        
    F_dhl=sum([dot(dot(C(i,frame),gamma_h(dx,dv)),gamma_l(v_))*dx(i) for i in range(nphases)]) 
    a3=form(F_dhl) 
    Dhl=assemble_matrix(a3)
    Dhl.assemble()   
    ai, aj, av=Dhl.getValuesCSR()
    Dhl=csr_matrix((av, aj, ai))    

    Dle_csr =csr_matrix(Dle)
    #DhlV0=np.matmul(Dhl.T,V0)
    DhlV0=Dhl.T.dot(V0_csr) 
        
    #DhlTV0Dle=np.matmul(Dhl,V0)+Dle
    DhlTV0Dle=Dhl.dot(V0_csr)+Dle_csr
        
    #V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)
    V0DllV0=(V0_csr.T.dot(Dll)).dot(V0_csr)
    
    # V1s
    bb=(DhlTV0Dle-DhlV0).toarray()
    for i in range(4):
        F=petsc4py.PETSc.Vec().createWithArray(bb[:,i],comm=MPI.COMM_WORLD)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
        nullspace_b.remove(F)
        V1s[:,i]= ksp_solve(A_l, F, V_l)
        
    V1s_csr=csr_matrix(V1s)    
    # Ainv
    Ainv=np.linalg.inv(D_eff).astype(np.float64)
    
    # B_tim
    B_tim=DhlTV0Dle.T.dot(V0_csr)
    B_tim=B_tim.toarray().astype(np.float64)
    
    # C_tim
    C_tim= V0DllV0 + V1s_csr.T.dot(DhlV0 + DhlTV0Dle) 
    C_tim=0.5*(C_tim+C_tim.T)
    C_tim=C_tim.toarray().astype(np.float64)
    
    # Ginv
    Q_tim=np.matmul(Ainv,np.array([(0,0),(0,0),(0,-1),(1,0)])).astype(np.float64)
    Ginv= np.matmul(np.matmul(Q_tim.T,(C_tim-np.matmul(np.matmul(B_tim.T,Ainv),B_tim))),Q_tim).astype(np.float64)
    G_tim=np.linalg.inv(Ginv)
    Y_tim= np.matmul(np.matmul(B_tim.T,Q_tim),G_tim)
    A_tim= D_eff + np.matmul(np.matmul(Y_tim,Ginv),Y_tim.T)

    # Deff_srt
    D=np.zeros((6,6))
    
    D[4:6,4:6]=G_tim
    D[0:4,4:6]=Y_tim
    D[4:6,0:4]=Y_tim.T
    D[0:4,0:4]=A_tim
    
    Deff_srt=np.zeros((6,6))
    Deff_srt[0,3:6]=A_tim[0,1:4]
    Deff_srt[0,1:3]=Y_tim[0,:]
    Deff_srt[0,0]=A_tim[0,0]
    
    Deff_srt[3:6,3:6]=A_tim[1:4,1:4]
    Deff_srt[3:6,1:3]=Y_tim[1:4,:]
    Deff_srt[3:6,0]=A_tim[1:4,0].flatten()
    
    Deff_srt[1:3,1:3]=G_tim
    Deff_srt[1:3,3:6]=Y_tim.T[:,1:4]
    Deff_srt[1:3,0]=Y_tim.T[:,0].flatten()
     
def initialize_array(V_l):
    xxx=3*V.dofmap.index_map.local_range[1]  # total dofs 
    V0 = np.zeros((xxx,4))
    Dle=np.zeros((xxx,4))
    Dhe=np.zeros((xxx,4)) 
    Dee=np.zeros((4,4)) 
    V1s=np.zeros((xxx,4))
    return V0,Dle,Dhe,Dee,V1s


def dof_mapping_quad(V, v2a, V_l, w_ll, boundary_facets_left, entity_mapl):
    """dof mapping makes solved unknown value w_l(Function(V_l)) assigned to v2a (Function(V)). 
    The boundary of wind blade mesh is a 1D curve. The facet/edge number is obtained from cell to edge connectivity (conn3) showed in subdomain subroutine.
    The same facet/edge number of extracted mesh_l (submesh) is obtaine din entity_mapl (gloabl mesh number). refer how submesh was generated.
    Therefore, once identifying the edge number being same for global(mesh)&boundary mesh(mesh_l), we equate the dofs and store w_l to v2a.
    The dofs can be verified by comparing the coordinates of local and global dofs if required. 

    Parameters
    ----------
    V : _type_
        _description_
    v2a : _type_
        _description_
    V_l : _type_
        _description_
    w_ll : 1D array (len(4))
        Fluctuating function data for case p
    boundary_facets_left : _type_
        _description_
    entity_mapl : _type_
        _description_


    Returns
    -------
    _type_
        _description_
    """
    dof_S2L = []
    deg=1
    for i,xx in enumerate(entity_mapl):
        dofs = locate_dofs_topological(V, 2, np.array([xx]))
        dofs_left= locate_dofs_topological(V_l, 2, np.array([boundary_facets_left[i]]))
        
        for k in range(deg+1):  
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.x.array[3*dofs[k]+j] = w_ll[3*dofs_left[k]+j] # store boundary solution of fluctuating functions
    return v2a
