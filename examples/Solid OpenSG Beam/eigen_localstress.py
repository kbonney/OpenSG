import opensg
import numpy as np
import time

taper_timo,left_timo,right_timo=[],[],[]
ar_time=[]
total_segments=28
beam_force=opensg.beam_reaction('bar_urc_npl_2_ar_4-segment_',total_segments)

for segment in np.linspace(0,0,1):
    tic = time.time() 
    print("\n Computing Segment:",str(int(segment))," \n")

    file_name='bar_urc_npl_2_ar_4-segment_'
    mesh_yaml = file_name+ str(int(segment)) +'.yaml'  ## the name of the yaml file containing the whole blade mesh
    mesh_data = opensg.load_yaml(mesh_yaml)
  
   # Blade Segment Mesh 
    blade_mesh = opensg.BladeMesh(mesh_data)  
    segment_mesh=blade_mesh.generate_segment_mesh(segment_index=0, filename="section.msh")
    print('\n Mesh Time of:'+str(blade_mesh.num_elements),"elements segment is ", time.time()-tic)
  
    # Homogenization
    timo=opensg.compute_stiffness(segment_mesh.material_database[0],
                                    segment_mesh.meshdata,
                                    segment_mesh.left_submesh,
                                    segment_mesh.right_submesh)
    # Dehomogenization
    strain_3D=opensg.recover_local_strain(timo,beam_force,segment,segment_mesh.meshdata) # Local Strain

    # Local Stress Path 
    file_name='solid.lp_sparcap_center_thickness_001' 
    points=np.loadtxt(file_name, delimiter=',', skiprows=0, dtype=str) # Load path coordinates
    eval_data=opensg.local_stress(segment_mesh.material_database[0],segment_mesh, strain_3D,points)
    for p in range(len(points)):
        print('Point:',[float(i) for i in points[p].split()],'   Stress Vector:', eval_data[p])
      
    # Eigen Solver
    eigen= opensg.eigen_stiffness_matrix(segment_mesh.material_database[0],segment_mesh,strain_3D, 2)

    print('\n Total Time for Segment',str(int(segment)),' :',time.time()-tic)
    
    
