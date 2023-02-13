#include "main.h"
#include "MeshDeformation.h"
#include "OpenMesh.h"
#include "OpenMesh/Tools/Decimater/DecimaterT.hh"
#include "OpenMesh/Tools/Decimater/ModQuadricT.hh"
#include "OpenMesh/Tools/Decimater/Observer.hh"

typedef Decimater::DecimaterT<SimpleMesh>  ADecimater;
typedef Decimater::ModQuadricT<SimpleMesh>::Handle HModQuadric;

//int main(int argc, const char * argv[])
int main()
{
    
    //if (argc >= 2) {
    //	filename = argv[1];
    //}

    std::string filename1 = "meshes/dancers-mesh_unwrapped-00000047.ply";
    std::string filename2 = "meshes/dancers-mesh_unwrapped-00000048.ply";

    std::vector<int>				constraintsIdx;
    std::vector<std::vector<float>> constraintsTarget;

    SimpleMesh* mesh1 = new SimpleMesh();
    SimpleMesh* mesh2 = new SimpleMesh();

    if (!OpenMesh::IO::read_mesh(*mesh1, filename1))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename1 << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", mesh1->n_faces(), mesh1->n_vertices());

    if (!OpenMesh::IO::read_mesh(*mesh2, filename2))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename2 << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", mesh2->n_faces(), mesh2->n_vertices());

    SimpleMesh* EDNodes = new SimpleMesh(*mesh1);
    ADecimater decimater(*EDNodes);
    HModQuadric hModQuadric;      
    decimater.add(hModQuadric);
    decimater.module(hModQuadric).unset_max_err();
    decimater.initialize();
    decimater.decimate_to_faces(1000,1000);
    EDNodes->garbage_collection();
    OpenMesh::IO::write_mesh(*EDNodes, "desimation.ply");
    MeshDeformation deform(mesh1, mesh2, EDNodes);
    SimpleMesh* res = deform.solve();

    if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << "out.off" << std::endl;
        exit(1);
    }
    printf("Saved\n");

#ifdef _WIN32
    getchar();
#endif
    return 0;
}
