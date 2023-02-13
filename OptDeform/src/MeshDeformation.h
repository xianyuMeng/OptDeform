#pragma once

#include <iostream>
#include <cuda_runtime.h>

#include "TerraWarpingSolver.h"
#include "OpenMesh.h"

class MeshDeformation
{
    public:
        MeshDeformation(const SimpleMesh* mesh, const SimpleMesh* target, const SimpleMesh* EDNodes)
        {
            m_result = *EDNodes;
            m_initial = *EDNodes;
            m_target = *target;
            m_origin = *mesh;
            m_stage = *mesh;

            unsigned int N = (unsigned int)EDNodes->n_vertices();
            unsigned int K = (unsigned int)mesh->n_vertices();
            unsigned int E = (unsigned int)EDNodes->n_edges();

            cutilSafeCall(cudaMalloc(&d_EDNodePos, sizeof(float3)*N));
            cutilSafeCall(cudaMalloc(&d_EDNodePosOrig, sizeof(float3)*N));
            cutilSafeCall(cudaMalloc(&d_vertexPosTarget, sizeof(float3)*K));
            cutilSafeCall(cudaMalloc(&d_vertexPosOrig, sizeof(float3)*K));

            cutilSafeCall(cudaMalloc(&d_rotMatrixLeft, sizeof(float3)*N));
            cutilSafeCall(cudaMalloc(&d_rotMatrixMiddle, sizeof(float3)*N));
            cutilSafeCall(cudaMalloc(&d_rotMatrixRight, sizeof(float3)*N));

            cutilSafeCall(cudaMalloc(&d_neighborEDNodeIdx0, sizeof(int)*K));
            cutilSafeCall(cudaMalloc(&d_neighborEDNodeIdx1, sizeof(int)*K));
            cutilSafeCall(cudaMalloc(&d_neighborEDNodeIdx2, sizeof(int)*K));
            cutilSafeCall(cudaMalloc(&d_neighborEDNodeIdx3, sizeof(int)*K));
            cutilSafeCall(cudaMalloc(&d_neighborEDNodeIdx4, sizeof(int)*K));
            
        
            
            h_neighbourIdx = new int[2 * E];
            h_neighbourOffset = new int[N + 1];
            h_neighborEDNodeIdx0 = new int[K];
            h_neighborEDNodeIdx1 = new int[K];
            h_neighborEDNodeIdx2 = new int[K];
            h_neighborEDNodeIdx3 = new int[K];
            h_neighborEDNodeIdx4 = new int[K];
            
           
            resetGPUMemory();
            
            m_optWarpingSolver = new TerraWarpingSolver(K, N, 2 * E, h_neighbourIdx, h_neighbourOffset, "MeshDeformationAD.t", "gaussNewtonGPU");

            

        } 

        void skinToEDNodes()
        {
            unsigned int N1 = (unsigned int)m_origin.n_vertices();
            unsigned int N2 = (unsigned int)m_initial.n_vertices();
            for (unsigned int i = 0; i < N1; i++)
            {
                const Vec3f& pt1 = m_origin.point(VertexHandle(i));
                float min_dist[4] = { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
                int nearestIdx[4];
                for (unsigned int j = 0; j < N2; j++)
                {
                    const Vec3f& pt2 = m_initial.point(VertexHandle(j));
                    float distance = (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]) + (pt1[2] - pt2[2]) * (pt1[2] - pt2[2]);
                    
                    if (min_dist[0] > distance || min_dist[1] > distance || min_dist[2] > distance || min_dist[3] > distance)
                    {
                        if (min_dist[0] >= min_dist[1] && min_dist[0] >= min_dist[2] && min_dist[0] >= min_dist[3])
                        {
                            min_dist[0] = distance;
                            nearestIdx[0] = j;

                        }
                        else if (min_dist[1] >= min_dist[0] && min_dist[1] >= min_dist[2] && min_dist[1] >= min_dist[3])
                        {
                            min_dist[1] = distance;
                            nearestIdx[1] = j;
                        }
                        else if (min_dist[2] >= min_dist[0] && min_dist[2] >= min_dist[1] && min_dist[2] >= min_dist[3])
                        {
                            min_dist[2] = distance;
                            nearestIdx[2] = j;
                        }
                        else if (min_dist[3] >= min_dist[0] && min_dist[3] >= min_dist[1] && min_dist[3] >= min_dist[2])
                        {
                            min_dist[3] = distance;
                            nearestIdx[3] = j;
                        }
                        else std::cout << "Wrong!" << std::endl;

                    }
                    

                }
                h_neighborEDNodeIdx0[i] = i;
                h_neighborEDNodeIdx1[i] = nearestIdx[0];
                h_neighborEDNodeIdx2[i] = nearestIdx[1];
                h_neighborEDNodeIdx3[i] = nearestIdx[2];
                h_neighborEDNodeIdx4[i] = nearestIdx[3];
            }
        }

        float dot(const Vec3f& v1, const Vec3f& v2)
        {
            return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        }

        Vec3f rotate(const Vec3f& RotLeft, const Vec3f& RotMiddle, const Vec3f& RotRight, const Vec3f& vector)
        {
            return Vec3f(dot(RotLeft,vector), dot(RotMiddle,vector), dot(RotRight,vector));
        }

        void calcVerticesPos()
        {
            unsigned int N = (unsigned int)m_result.n_vertices();
            
            float3* h_rotMatrixLeft = new float3[N];
            float3* h_rotMatrixMiddle = new float3[N];
            float3* h_rotMatrixRight = new float3[N];
            cutilSafeCall(cudaMemcpy(h_rotMatrixLeft, d_rotMatrixLeft, sizeof(float3)*N, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_rotMatrixMiddle, d_rotMatrixMiddle, sizeof(float3)*N, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_rotMatrixRight, d_rotMatrixRight, sizeof(float3)*N, cudaMemcpyDeviceToHost));

            unsigned int K = (unsigned int)m_origin.n_vertices();
            for (unsigned int i = 0; i < K; i++)
            {
                const Vec3f& pt = m_origin.point(VertexHandle(i));
                unsigned int idx1 = h_neighborEDNodeIdx1[i];
                unsigned int idx2 = h_neighborEDNodeIdx2[i];
                unsigned int idx3 = h_neighborEDNodeIdx3[i];
                unsigned int idx4 = h_neighborEDNodeIdx4[i];

                const Vec3f& EDNode1 = m_initial.point(VertexHandle(idx1));
                const Vec3f& EDNode2 = m_initial.point(VertexHandle(idx2));
                const Vec3f& EDNode3 = m_initial.point(VertexHandle(idx3));
                const Vec3f& EDNode4 = m_initial.point(VertexHandle(idx4));

                const Vec3f& Offset1 = m_result.point(VertexHandle(idx1));
                const Vec3f& Offset2 = m_result.point(VertexHandle(idx2));
                const Vec3f& Offset3 = m_result.point(VertexHandle(idx3));
                const Vec3f& Offset4 = m_result.point(VertexHandle(idx4));

                Vec3f RotLeft1 = Vec3f(h_rotMatrixLeft[idx1].x, h_rotMatrixLeft[idx1].y, h_rotMatrixLeft[idx1].z);
                Vec3f RotLeft2 = Vec3f(h_rotMatrixLeft[idx2].x, h_rotMatrixLeft[idx2].y, h_rotMatrixLeft[idx2].z);
                Vec3f RotLeft3 = Vec3f(h_rotMatrixLeft[idx3].x, h_rotMatrixLeft[idx3].y, h_rotMatrixLeft[idx3].z);
                Vec3f RotLeft4 = Vec3f(h_rotMatrixLeft[idx4].x, h_rotMatrixLeft[idx4].y, h_rotMatrixLeft[idx4].z);

                Vec3f RotRight1 = Vec3f(h_rotMatrixRight[idx1].x, h_rotMatrixRight[idx1].y, h_rotMatrixRight[idx1].z);
                Vec3f RotRight2 = Vec3f(h_rotMatrixRight[idx2].x, h_rotMatrixRight[idx2].y, h_rotMatrixRight[idx2].z);
                Vec3f RotRight3 = Vec3f(h_rotMatrixRight[idx3].x, h_rotMatrixRight[idx3].y, h_rotMatrixRight[idx3].z);
                Vec3f RotRight4 = Vec3f(h_rotMatrixRight[idx4].x, h_rotMatrixRight[idx4].y, h_rotMatrixRight[idx4].z);

                Vec3f RotMiddle1 = Vec3f(h_rotMatrixMiddle[idx1].x, h_rotMatrixMiddle[idx1].y, h_rotMatrixMiddle[idx1].z);
                Vec3f RotMiddle2 = Vec3f(h_rotMatrixMiddle[idx2].x, h_rotMatrixMiddle[idx2].y, h_rotMatrixMiddle[idx2].z);
                Vec3f RotMiddle3 = Vec3f(h_rotMatrixMiddle[idx3].x, h_rotMatrixMiddle[idx3].y, h_rotMatrixMiddle[idx3].z);
                Vec3f RotMiddle4 = Vec3f(h_rotMatrixMiddle[idx4].x, h_rotMatrixMiddle[idx4].y, h_rotMatrixMiddle[idx4].z);


                Vec3f ptt = (rotate(RotLeft1, RotMiddle1, RotRight1, (pt - EDNode1)) + Offset1 +
                            rotate(RotLeft2, RotMiddle2, RotRight2, (pt - EDNode2)) + Offset2 +
                            rotate(RotLeft3, RotMiddle3, RotRight3, (pt - EDNode3)) + Offset3 + 
                            rotate(RotLeft4, RotMiddle4, RotRight4, (pt - EDNode4)) + Offset4) / 4;
                
                m_stage.set_point(VertexHandle(i), ptt);
            }

            
            delete[] h_rotMatrixLeft;
            delete[] h_rotMatrixMiddle;
            delete[] h_rotMatrixRight;
        }
        void findNearestPts()
        {
            m_constraintsIdx.clear();
            m_constraintsTarget.clear();
            //unsigned int N1 = (unsigned int)m_origin.n_vertices();
            unsigned int N1 = (unsigned int)m_stage.n_vertices();
            unsigned int N2 = (unsigned int)m_target.n_vertices();
            for (unsigned int i = 0; i < N1; i++)
            {
                const Vec3f& pt1 = m_stage.point(VertexHandle(i));
                
                
                float min_dist = std::numeric_limits<float>::infinity();
                Vec3f nearest;
                for (unsigned int j = 0; j < N2; j++)
                {
                    const Vec3f& pt2 = m_target.point(VertexHandle(j));
                    float distance = (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]) + (pt1[2] - pt2[2]) * (pt1[2] - pt2[2]);
                    if (min_dist > distance)
                    {
                        min_dist = distance;
                        nearest = pt2;
                    }
                }
                m_constraintsIdx.push_back(i);
                std::vector<float> point;
                point.push_back(nearest[0]);
                point.push_back(nearest[1]);
                point.push_back(nearest[2]);
                m_constraintsTarget.push_back(point);
            }
        }
        void setConstraints(float alpha)
        {
            unsigned int N = (unsigned int)m_origin.n_vertices();
            float3* h_vertexPosTargetFloat3 = new float3[N];
            for (unsigned int i = 0; i < N; i++)
            {
                h_vertexPosTargetFloat3[i] = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            }

            for (unsigned int i = 0; i < m_constraintsIdx.size(); i++)
            {
                const Vec3f& pt = m_origin.point(VertexHandle(m_constraintsIdx[i]));
                const Vec3f target = Vec3f(m_constraintsTarget[i][0], m_constraintsTarget[i][1], m_constraintsTarget[i][2]);

                Vec3f z = (1 - alpha)*pt + alpha*target;
                h_vertexPosTargetFloat3[m_constraintsIdx[i]] = make_float3(z[0], z[1], z[2]);
            }
            cutilSafeCall(cudaMemcpy(d_vertexPosTarget, h_vertexPosTargetFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
            delete [] h_vertexPosTargetFloat3;
        }


        void resetGPUMemory()
        {
            unsigned int N = (unsigned int)m_initial.n_vertices();
            unsigned int E = (unsigned int)m_initial.n_edges();
            unsigned int K = (unsigned int)m_origin.n_vertices();

            float3* h_EDNodePos = new float3[N];
            int*	h_numNeighbours   = new int[N];
            float3* h_vertexPosOrig = new float3[K];

            // Set EDNodes
            for (unsigned int i = 0; i < N; i++)
            {
                const Vec3f& pt = m_initial.point(VertexHandle(i));
                h_EDNodePos[i] = make_float3(pt[0], pt[1], pt[2]);
            }

            // Set Vertices
            for (unsigned int i = 0; i < K; i++)
            {
                const Vec3f& pt = m_origin.point(VertexHandle(i));
                h_vertexPosOrig[i] = make_float3(pt[0], pt[1], pt[2]);
            }

            // Skin to EDNodes
            skinToEDNodes();

            // Set Neighbour
            unsigned int count = 0;
            unsigned int offset = 0;
            h_neighbourOffset[0] = 0;
            for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
            {
                VertexHandle c_vh(v_it.handle());
                unsigned int valance = m_initial.valence(c_vh);
                h_numNeighbours[count] = valance;

                for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it; vv_it++)
                {
                    VertexHandle v_vh(vv_it.handle());

                    h_neighbourIdx[offset] = v_vh.idx();
                    offset++;
                }

                h_neighbourOffset[count + 1] = offset;

                count++;
            }
            
            // Find Nearest Points
            findNearestPts();
            setConstraints(1.0f);


            // Rotate Matrix
            float3* h_rotMatrixLeft = new float3[N];
            float3* h_rotMatrixMiddle = new float3[N];
            float3* h_rotMatrixRight = new float3[N];
            for (unsigned int i = 0; i < N; i++)
            {
                h_rotMatrixLeft[i] = make_float3(1.0f, 0.0f, 0.0f);
                h_rotMatrixMiddle[i] = make_float3(0.0f, 1.0f, 0.0f);
                h_rotMatrixRight[i] = make_float3(0.0f, 0.0f, 1.0f);
            }
            
            cutilSafeCall(cudaMemcpy(d_rotMatrixLeft, h_rotMatrixLeft, sizeof(float3)*N, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_rotMatrixMiddle, h_rotMatrixMiddle, sizeof(float3)*N, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_rotMatrixRight, h_rotMatrixRight, sizeof(float3)*N, cudaMemcpyHostToDevice));
         

            cutilSafeCall(cudaMemcpy(d_EDNodePos, h_EDNodePos, sizeof(float3)*N, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_EDNodePosOrig, h_EDNodePos, sizeof(float3)*N, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_vertexPosOrig, h_vertexPosOrig, sizeof(float3)*K, cudaMemcpyHostToDevice));

            cutilSafeCall(cudaMemcpy(d_neighborEDNodeIdx0, h_neighborEDNodeIdx0, sizeof(int)*K, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_neighborEDNodeIdx1, h_neighborEDNodeIdx1, sizeof(int)*K, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_neighborEDNodeIdx2, h_neighborEDNodeIdx2, sizeof(int)*K, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_neighborEDNodeIdx3, h_neighborEDNodeIdx3, sizeof(int)*K, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_neighborEDNodeIdx4, h_neighborEDNodeIdx4, sizeof(int)*K, cudaMemcpyHostToDevice));
            

            delete[] h_rotMatrixLeft;
            delete[] h_rotMatrixMiddle;
            delete[] h_rotMatrixRight;
            delete[] h_EDNodePos;
            delete[] h_numNeighbours;
            delete[] h_vertexPosOrig;
        }

        ~MeshDeformation()
        {
            delete[] h_neighbourIdx;
            delete[] h_neighbourOffset;
            delete[] h_neighborEDNodeIdx0;
            delete[] h_neighborEDNodeIdx1;
            delete[] h_neighborEDNodeIdx2;
            delete[] h_neighborEDNodeIdx3;
            delete[] h_neighborEDNodeIdx4;

            cutilSafeCall(cudaFree(d_EDNodePos));
            cutilSafeCall(cudaFree(d_EDNodePosOrig));
            cutilSafeCall(cudaFree(d_vertexPosOrig));
            cutilSafeCall(cudaFree(d_vertexPosTarget));

            cutilSafeCall(cudaFree(d_rotMatrixLeft));
            cutilSafeCall(cudaFree(d_rotMatrixMiddle));
            cutilSafeCall(cudaFree(d_rotMatrixRight));

            cutilSafeCall(cudaFree(d_neighborEDNodeIdx0));
            cutilSafeCall(cudaFree(d_neighborEDNodeIdx1));
            cutilSafeCall(cudaFree(d_neighborEDNodeIdx2));
            cutilSafeCall(cudaFree(d_neighborEDNodeIdx3));
            cutilSafeCall(cudaFree(d_neighborEDNodeIdx4));

            SAFE_DELETE(m_optWarpingSolver);
        }

        SimpleMesh* solve()
        {
            float weightFit = 10.0f;
            float weightReg = 100.0f; 
            float weightRot = 1.0f;
            unsigned int numIter = 20;
            unsigned int nonLinearIter = 10;
            unsigned int linearIter = 20;

            copyResultToCPUFromFloat3();

            m_result = m_initial;
            resetGPUMemory();
            for (unsigned int i = 1; i < numIter; i++)
            {
                std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
                
                unsigned int N = (unsigned int)m_result.n_vertices();
                float3* h_EDNodePos = new float3[N];
                cutilSafeCall(cudaMemcpy(h_EDNodePos, d_EDNodePos, sizeof(float3)*N, cudaMemcpyDeviceToHost));
                for (unsigned int i = 0; i < N; i++)
                {
                    m_result.set_point(VertexHandle(i), Vec3f(h_EDNodePos[i].x, h_EDNodePos[i].y, h_EDNodePos[i].z));
                }
                calcVerticesPos();
                findNearestPts();
                setConstraints(1.0f);
                
                //setConstraints((float)i / (float)(numIter - 1));
                
                m_optWarpingSolver->solveGN(d_EDNodePos, d_rotMatrixLeft, d_rotMatrixMiddle, d_rotMatrixRight, d_EDNodePosOrig, d_vertexPosTarget, d_vertexPosOrig,
                    d_neighborEDNodeIdx0, d_neighborEDNodeIdx1, d_neighborEDNodeIdx2, d_neighborEDNodeIdx3, d_neighborEDNodeIdx4, nonLinearIter, linearIter, weightFit, weightReg, weightRot);

                delete[] h_EDNodePos;
                
            }
            copyResultToCPUFromFloat3();
            calcVerticesPos();
            //return &m_result;
            return &m_stage;
        }

        void copyResultToCPUFromFloat3()
        {
            unsigned int N = (unsigned int)m_result.n_vertices();
            float3* h_EDNodePos = new float3[N];
            cutilSafeCall(cudaMemcpy(h_EDNodePos, d_EDNodePos, sizeof(float3)*N, cudaMemcpyDeviceToHost));


            for (unsigned int i = 0; i < N; i++)
            {
                m_result.set_point(VertexHandle(i), Vec3f(h_EDNodePos[i].x, h_EDNodePos[i].y, h_EDNodePos[i].z));
            }

            delete [] h_EDNodePos;
        }

    private:

        SimpleMesh m_result;
        SimpleMesh m_initial;
        SimpleMesh m_target;
        SimpleMesh m_origin;
        SimpleMesh m_stage;

        float3* d_rotMatrixLeft;
        float3* d_rotMatrixMiddle;
        float3* d_rotMatrixRight;

        
        float3*	d_EDNodePos;
        float3*	d_EDNodePosOrig;
        float3*	d_vertexPosTarget;
        float3* d_vertexPosOrig;

        int*	d_numNeighbours;

        int*    d_neighborEDNodeIdx0;
        int*    d_neighborEDNodeIdx1;
        int*    d_neighborEDNodeIdx2;
        int*    d_neighborEDNodeIdx3;
        int*    d_neighborEDNodeIdx4;
        int*    h_neighborEDNodeIdx0;
        int*    h_neighborEDNodeIdx1;
        int*    h_neighborEDNodeIdx2;
        int*    h_neighborEDNodeIdx3;
        int*    h_neighborEDNodeIdx4;

        int*	h_neighbourIdx;
        int*	h_neighbourOffset;

        TerraWarpingSolver* m_optWarpingSolver;

        std::vector<int>				m_constraintsIdx;
        std::vector<std::vector<float>>	m_constraintsTarget;

     

};
