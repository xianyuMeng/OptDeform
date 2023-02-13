#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <math.h>

#include "cudaUtil.h"

extern "C" {
#include "Opt.h"
}


template <class type> type* createDeviceBuffer(const std::vector<type>& v) {
    type* d_ptr;
    cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));

    cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
    return d_ptr;
}

class TerraWarpingSolver {

public:
    TerraWarpingSolver(unsigned int vertexCount, unsigned int EDNodeCount, unsigned int E, int* d_xCoords, int* d_offsets, const std::string& terraFile, const std::string& optName) :
        m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr), vertexCount(vertexCount)
    {
        edgeCount = (int)E;
        
        m_optimizerState = Opt_NewState();
        m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

        int* h_offsets = d_offsets;
        int* h_xCoords = d_xCoords;
        
        // Convert to our edge format
        std::vector<int> h_headX;
        std::vector<int> h_tailX;
        for (int headX = 0; headX < (int)EDNodeCount; ++headX) {
            for (int j = h_offsets[headX]; j < h_offsets[headX + 1]; ++j) {
                h_headX.push_back(headX);
                h_tailX.push_back(h_xCoords[j]);
            }
        }

        d_headX = createDeviceBuffer(h_headX);
        d_tailX = createDeviceBuffer(h_tailX);

        //uint32_t dims[] = { vertexCount, 1 };
        //unsigned int dims[] = { EDNodeCount, 1 };
        unsigned int dims[] = { vertexCount, EDNodeCount };
        m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

        assert(m_optimizerState);
        assert(m_problem);
        assert(m_plan);


        m_numUnknown = vertexCount;
    }

    ~TerraWarpingSolver()
    {
        cutilSafeCall(cudaFree(d_headX));
        
        cutilSafeCall(cudaFree(d_tailX));
        

        if (m_plan) {
            Opt_PlanFree(m_optimizerState, m_plan);
        }

        if (m_problem) {
            Opt_ProblemDelete(m_optimizerState, m_problem);
        }

    }

    void solveGN(
        float3* d_vertexPosFloat3,
        float3* d_rotMatrixLeft,
        float3* d_rotMatrixMiddle,
        float3* d_rotMatrixRight,
        float3* d_vertexPosFloat3Urshape,
        float3* d_vertexPosTargetFloat3,
        float3* d_vertexPosOrigFloat3,
        int* d_neighborEDNodeIdx0,
        int* d_neighborEDNodeIdx1,
        int* d_neighborEDNodeIdx2,
        int* d_neighborEDNodeIdx3,
        int* d_neighborEDNodeIdx4,
        unsigned int nNonLinearIterations,
        unsigned int nLinearIterations,
        float weightFit,
        float weightReg,
        float weightRot)
    {
        unsigned int nBlockIterations = 1;	//invalid just as a dummy;

        void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

        float weightFitSqrt = sqrt(weightFit);
        float weightRegSqrt = sqrt(weightReg);
        float weightRotSqrt = sqrt(weightRot);
        
        
        //void* problemParams[] = { &weightFitSqrt, &weightRegSqrt, &weightRotSqrt, d_vertexPosFloat3, d_rotMatrixLeft, d_rotMatrixMiddle, d_rotMatrixRight, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, &edgeCount, d_headX, d_tailX};

        
        void* problemParams[] = { &weightFitSqrt, &weightRegSqrt, &weightRotSqrt, d_vertexPosFloat3, d_rotMatrixLeft, d_rotMatrixMiddle, d_rotMatrixRight, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, d_vertexPosOrigFloat3, &edgeCount, d_headX, d_tailX,
            &vertexCount, d_neighborEDNodeIdx0, d_neighborEDNodeIdx1, d_neighborEDNodeIdx2, d_neighborEDNodeIdx3, d_neighborEDNodeIdx4 };
        
    

        Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
    }

private:

    int* d_headX;
    int* d_tailX;


    int edgeCount;
    int vertexCount;


    Opt_State*		m_optimizerState;
    Opt_Problem*	m_problem;
    Opt_Plan*		m_plan;

    unsigned int m_numUnknown;
};
