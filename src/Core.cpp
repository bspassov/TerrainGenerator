#pragma once
#include "mat.h"
#include "Angel.h"
#include <cuda_runtime.h>
#include "cutil_math.h"
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include "PerlinNoise.h"

//Types

typedef Angel::vec4  color4;
typedef Angel::vec4  point4;



struct Camera
{
	vec4 pos;
	vec4 dir;
	vec4 side;
	GLfloat phi;
	GLfloat theta;
};

//Data
//Constants go here
namespace C
{	
	//Used to initialize the static memory arrays' sizes
	//521*512=262144
	const int GRID_SIDE_MAX = 515;
	const int GRID_VERTEX_CNT_MAX = GRID_SIDE_MAX*GRID_SIDE_MAX; 
	const int GRID_TRI_MAX = (GRID_SIDE_MAX-1)*(GRID_SIDE_MAX-1)*6;
	const GLfloat CAM_ROT_FACTOR_X = .5;
	const GLfloat CAM_ROT_FACTOR_Y = .5;
	const GLfloat CAM_MOV_SPEED = 44.0;
}

// Various globals go here
namespace G
{
	PerlinNoise noiseGenerator;
	int iterations = 0;
	int time_Current = 0;
	int time_Previous = 0;	
	float dt;
	float timestep;

	//Geometry Stuff
	point4 gridVectorPositions[512*512*2];
	point4 gridMeshPositions[C::GRID_VERTEX_CNT_MAX];
	GLuint gridMeshIndices[C::GRID_TRI_MAX];
	int gridSideCnt = 0;
	int gridVertexCnt = 0;
	int gridTriCnt = 0;
	//---------------------------------------------------------------
	
	//Input Stuff
	int uiMouseNewX = 0;
	int uiMouseNewY = 0;
	int uiMouseOldX = 0;
	int uiMouseOldY = 0;
	vec3 uimouseWorld;
	int uiMouse_L = 0;
	int uiMouse_R = 0;
	int uiMouse_M = 0;
	
	int uiKey_W = 0;
	int uiKey_A = 0;
	int uiKey_S = 0;
	int uiKey_D = 0;
	int uiKey_Q = 0;
	int uiKey_E = 0;
	int uiKey_R = 0;
	int uiKey_P = 0;
	int uiKey_Space = 0;

	float uiEditRadius =15.0f;
	//---------------------------------------------------------------
	

	//Window Stuff
	char uiwinTitle[20];
	GLfloat uiwinX;
	GLfloat uiwinY;
	//---------------------------------------------------------------

	//Camera Stuff
	Camera uiCamera;
	GLfloat uiFovy = 60.0f;  // Field-of-view in Y direction angle (in degrees)
	GLfloat uiAspect = 1.0f;       // Viewport aspect ratio
	GLfloat uizNear = 0.05f;
	GLfloat	uizFar = 2900.0f;
	mat4 mv;
	mat4 p;
	//---------------------------------------------------------------
	
	//Shader Stuff
	GLuint shaderGround;
	GLuint shaderVPosition;
	GLuint shaderModel_view;  // model-view matrix uniform shader variable location
	GLuint shaderMousePosition;  // model-view matrix uniform shader variable location
	GLuint shaderProjection; // projection matrix uniform shader variable location
	GLuint shaderWorldScale;
	GLuint shaderEditDistance;
	GLuint tex_loc ;
	GLuint vShader_tex_loc;
	GLuint type;
	//---------------------------------------------------------------
	
	//CUDA-GL comunications
	//Initial Terrain Height;
	float initialTerrainH[C::GRID_VERTEX_CNT_MAX];
	float initialRainRate[C::GRID_VERTEX_CNT_MAX];
	float initialHardness[C::GRID_VERTEX_CNT_MAX];

	GLuint pboGroundH;
	GLuint texGroundH;
	
	//Cuda
	cudaError cuda_ERR;
	//

	//Pointers to constant memory stuff
	int *hCoef_TimestepPtr;
	float hCoef_Timestep = 0.05f;
	int *hCoef_GPtr;
	float hCoef_G = 9.81f;
	int *hCoef_PipeCrossSectionPtr;
	float hCoef_PipeCrossSection = 1.0f;
	int *hCoef_PipeLengthPtr;
	float hCoef_PipeLength = 1.0f;

	int *hCoef_RainRatePtr;
	float hCoef_RainRate = 0.18;//0.180;//.18;//.192;

	int *hCoef_talusRatioPtr;
	float hCoef_talusRatio = 0.8f;//0.8;
	int *hCoef_talusCoefPtr;
	float hCoef_talusCoef = 1.7f;
	int *hCoef_talusBiasPtr;
	float hCoef_talusBias = 0.1f;

	int *hCoef_SedimentCapacityFactorPtr;
	float hCoef_SedimentCapacityFactor = 10.0f;
	int *hCoef_DepthMaxPtr;
	float hCoef_DepthMax = 2.5f;

	int *hCoef_DissolveRatePtr;
	float hCoef_DissolveRate = .50f;//0.;//5.5;
	int *hCoef_SedimentDropRatePtr;
	float hCoef_SedimentDropRate = .640f;//1.4;//0.;//5.4;
	int *hCoef_HardnessMinPtr;
	float hCoef_HardnessMin = 0.5f;
	int *hCoef_SoftenRatePtr;
	float hCoef_SoftenRate = .2f;

	int *hCoef_ThermalErosionRatePtr;
	float hCoef_ThermalErosionRate = 3.0f;//5.0;//05.150;// 0.165;

	int *hCoef_EvaporationRatePtr;
	float hCoef_EvaporationRate = 0.05f;//0.015;
}
//-------------------CUDA-----------------------------------
//Externs
extern "C" void cuda_Initialize(int width, int height);
extern "C" int cuda_SetTerrainHeight(void * src, size_t srcPitch,size_t width, size_t height);
extern "C" int cuda_SetHardness(void * src, size_t srcPitch,size_t width, size_t height);
extern "C" int cuda_SetRainRate(void * src, size_t srcPitch,size_t width, size_t height);
extern "C" void cuda_Simulate(vec4* heights, unsigned int width, unsigned int height, float dt, float * in,float* in2);
extern "C" void cuda_EditTerrain(float editX, float editY, float pointDistance, float editValue, unsigned int width, unsigned int height ,float maxDist, float dt , int mode);
extern "C" float* cuda_fetchTerrainHptr();
extern "C" float* cuda_fetchWaterHptr();
extern "C" float* cuda_fetchHardnessptr();
extern "C" float* cuda_fetchRainRateptr();
extern "C" float* cuda_fetchWaterVelocityPtr();
extern "C" float* cuda_fetchSedimentAmountPtr();
//Others
cudaGraphicsResource* CUDA_pboGroundH_ptr;
vec4* heights_ptr;
//----------------------------------------------------------
GLuint buffer;
GLuint indexBuffer;
vec4 tester[512*512];
vec3 CollisionTest;
void frustumPlanes()
{
	mat4 frustum=G::p*G::mv;
	vec4 planes[6];
	for(int i=0;i<3;i++)
	{
		planes[i*2]=frustum[3]+frustum[i];
		planes[i*2+1]=frustum[3]-frustum[i];
	}
	int sentinel=0;
	for (int i=0;i<6;i++)
	{
		int temp = ( dot( vec3(2.5,40.0,-2.5), vec3( planes[i].x,planes[i].y,planes[i].z ) ) + 
					 dot( vec3(2.5,0.0,2.5),   vec3( abs(planes[i].x),abs(planes[i].y),abs(planes[i].z) ) ) > -planes[i].w);
		sentinel|=temp<<i;		
	}
	//printf("\n\nInside: %i %X \n\n",sentinel==63, sentinel);
}

void generate_heightfield_heights(point4 *positions, float *heights,  int width, int height, int detail, float amplitude, float scale)
{
	//Makes sure that the height isn't less than 0;
	float bias = amplitude*0.5f;
	//DEBUG
	printf("\nGenerating Height Positions [|____________]\b\b\b\b\b\b\b\b\b\b\b\b");	
	G::noiseGenerator.setSeed(rand());
	float h=0.0f;
	for (int i=0;i<height;i++)
	{
		
		for (int j=0;j<width;j++)
		{
			h=(G::noiseGenerator.fbm(i*0.002f,j*0.002f,0.0f,detail,0.50f,1.95f)*amplitude)*scale;
			h*=h;
			//h=abs(i-256)*0.25;
			positions[i*width+j].y = 0.0;
			heights[i*width+j] = h;
		}
		//DEBUG
		if (i%(height/10)==0) printf("\b=|");
	}
}


void generate_heightfield_hardness(float *hardness, int width, int height, int detail,float minHardness, float maxHardness)
{
	//DEBUG
	printf("\nGenerating Hardness Values[|____________]\b\b\b\b\b\b\b\b\b\b\b\b");

	G::noiseGenerator.setSeed(rand());	
	for (int i=0;i<height;i++)
	{
		
		for (int j=0;j<width;j++)
		{
			hardness[i*width+j] = minHardness+abs(G::noiseGenerator.fbm(i*0.002,j*0.002,0.0,detail,0.60,1.95))*(maxHardness-minHardness);			
			//hardness[i*width+j] = 0.0;
		}
		//DEBUG
		if (i%(height/10)==0) printf("\b=|");
	}
}

void generate_heightfield_rain(float *rain, int width, int height, int detail,float minRain, float maxRain)
{
	//DEBUG
	printf("\nGenerating Rain Values[|____________]\b\b\b\b\b\b\b\b\b\b\b\b");

	G::noiseGenerator.setSeed(rand());	
	for (int i=0;i<height;i++)
	{
		
		for (int j=0;j<width;j++)
		{
            float noise = G::noiseGenerator.fbm(i*0.02,j*0.02,0.0,detail,0.60,1.95);
            float temp = fmaxf(0.0,minRain+abs(noise)*(maxRain-minRain)-0.4);			
			rain[i*width+j] = temp;
            
		}
		//DEBUG
		if (i%(height/10)==0) printf("\b=|");
	}
}

void generate_heightfield_mesh(point4 *positions,GLuint *indices, int width,int height,float distance)
{
	//Generate Positions
		
	for (int i=0;i<height;i++)
	{
		
		for (int j=0;j<width;j++)
		{
			float x = (j-height*0.5+0.5)*distance;
			float z = -(i-width*0.5+0.5)*distance;
			
			positions[i*width+j]=point4(x,0.0,z,1.0);	
		}
		
	}
	
	GLuint index=0;
	for(int u=1;u<height;u++)
	{
		for(int v=1;v<width;v++)
		{
			if ((u^v)&1) //one index is odd,one index is even
			{
				//First triangle
				indices[index] = (u-1)*width+v;
				index++;
				indices[index] = (u-1)*width+v-1;
				index++;
				indices[index] = u*width+v-1;
				index++;
				//Second triangle
				indices[index] = (u-1)*width+v;
				index++;
				indices[index] = u*width+v-1;
				index++;
				indices[index] = u*width+v;
				index++;
			}
			else
			{				
				//First triangle
				indices[index] = (u-1)*width+v-1;
				index++;
				indices[index] = u*width+v-1;
				index++;
				indices[index] = u*width+v;
				index++;
				//Second triangle
				indices[index] = (u-1)*width+v-1;
				index++;
				indices[index] = u*width+v;
				index++;
				indices[index] = (u-1)*width+v;
				index++;
			}	
		}
	}
}

void generate_vectorField_mesh(point4 *positions, int width,int height,float distance)
{
	//Generate Positions		
	for (int i=0;i<height;i++)
	{		
		for (int j=0;j<width;j++)
		{
			float x = (j-height*0.5+0.5)*distance;
			float z = -(i-width*0.5+0.5)*distance;
			
			positions[i*width*2+j*2]=point4(x,0.0,z,1.0);				
			positions[i*width*2+j*2+1]=point4(x,1.0,z,1.0);
		}		
	}	
}


float *dMain;
float *dMain2;
float *dMainAr;
float checker[512*512];
	
// OpenGL initialization 
int first=1;
int step=0;

void runCUDA()
{
	//if(!first)
	//{
	//float * ptrArray = cuda_fetchRainRateptr();
	//G::cuda_ERR = cudaGetLastError();	
	//G::cuda_ERR=cudaMemcpy2D(ptrArray, 2048, &checker[0], 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyHostToDevice);	
	//}
	size_t num_bytes=G::gridSideCnt*G::gridSideCnt*sizeof(float4);
	G::cuda_ERR = cudaGetLastError();
    G::cuda_ERR = cudaGraphicsMapResources(1, &CUDA_pboGroundH_ptr, 0);   
	 
    G::cuda_ERR = cudaGraphicsResourceGetMappedPointer((void**)&heights_ptr, &num_bytes, CUDA_pboGroundH_ptr);
	
	if(G::uiKey_P)
	{step^=1;}
	if((step&&G::uiKey_E)||!step)
	{
		cuda_Simulate(heights_ptr, G::gridSideCnt, G::gridSideCnt,float(G::time_Current)/1000.f, dMainAr, dMain2);
		
	}
	else printf("\n Sim PAUSED \n");

	G::cuda_ERR = cudaGetLastError();
	G::cuda_ERR = cudaGraphicsUnmapResources(1, &CUDA_pboGroundH_ptr, 0); 
	//G::cuda_ERR = cudaMemcpy(&blah,dMain,4,cudaMemcpyDeviceToHost);
	//G::cuda_ERR = cudaMemcpy(&blah2,dMain2,4,cudaMemcpyDeviceToHost);
	//G::cuda_ERR = cudaMemcpy(&checker[0],dMainAr,4*512*512,cudaMemcpyDeviceToHost);
	first=0;
}

void initCUDA()
{

	
	//G::cuda_ERR = cudaMemcpy(dMain,&blah,4*512*512,cudaMemcpyHostToDevice)
	
	//G::cuda_ERR = cudaDeviceReset();
	cuda_Initialize(G::gridSideCnt, G::gridSideCnt);
	G::cuda_ERR =cudaGetLastError(); 
	//Registration of buffers
	G::cuda_ERR = cudaGraphicsGLRegisterBuffer((cudaGraphicsResource **)&CUDA_pboGroundH_ptr, G::pboGroundH, cudaGraphicsRegisterFlagsNone );
	//printf("\n!!!!\t\t!!!!\t %i",G::cuda_ERR);
	G::cuda_ERR =cudaGetLastError();
	//Registration of constants
	cudaGetSymbolAddress((void **)&G::hCoef_TimestepPtr, "dCoef_Timestep");
	cudaGetSymbolAddress((void **)&G::hCoef_GPtr, "dCoef_G");
	cudaGetSymbolAddress((void **)&G::hCoef_PipeCrossSectionPtr, "dCoef_PipeCrossSection");
	cudaGetSymbolAddress((void **)&G::hCoef_PipeLengthPtr, "dCoef_PipeLength");
	cudaGetSymbolAddress((void **)&G::hCoef_RainRatePtr, "dCoef_RainRate");
	cudaGetSymbolAddress((void **)&G::hCoef_talusRatioPtr, "dCoef_talusRatio");
	cudaGetSymbolAddress((void **)&G::hCoef_talusCoefPtr, "dCoef_talusCoef");
	cudaGetSymbolAddress((void **)&G::hCoef_talusBiasPtr, "dCoef_talusBias");
	cudaGetSymbolAddress((void **)&G::hCoef_SedimentCapacityFactorPtr, "dCoef_SedimentCapacityFactor");
	cudaGetSymbolAddress((void **)&G::hCoef_DepthMaxPtr, "dCoef_DepthMax");
	cudaGetSymbolAddress((void **)&G::hCoef_DissolveRatePtr, "dCoef_DissolveRate");
	cudaGetSymbolAddress((void **)&G::hCoef_SedimentDropRatePtr, "dCoef_SedimentDropRate");
	cudaGetSymbolAddress((void **)&G::hCoef_HardnessMinPtr, "dCoef_HardnessMin");
	cudaGetSymbolAddress((void **)&G::hCoef_SoftenRatePtr, "dCoef_SoftenRate");
	cudaGetSymbolAddress((void **)&G::hCoef_ThermalErosionRatePtr, "dCoef_ThermalErosionRate");
	cudaGetSymbolAddress((void **)&G::hCoef_EvaporationRatePtr, "dCoef_EvaporationRate");
	
	//Initialization of constants
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_TimestepPtr, (void const *) &G::hCoef_Timestep,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_GPtr, (void const *) &G::hCoef_G,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_PipeCrossSectionPtr, (void const *) &G::hCoef_PipeCrossSection,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_PipeLengthPtr, (void const *) &G::hCoef_PipeLength,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_RainRatePtr, (void const *) &G::hCoef_RainRate,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_talusRatioPtr, (void const *) &G::hCoef_talusRatio,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_talusCoefPtr, (void const *) &G::hCoef_talusCoef,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_talusBiasPtr, (void const *) &G::hCoef_talusBias,sizeof(float),cudaMemcpyHostToDevice);	
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_SedimentCapacityFactorPtr, (void const *) &G::hCoef_SedimentCapacityFactor,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_DepthMaxPtr, (void const *) &G::hCoef_DepthMax,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_DissolveRatePtr, (void const *) &G::hCoef_DissolveRate,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_SedimentDropRatePtr, (void const *) &G::hCoef_SedimentDropRate,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_HardnessMinPtr, (void const *) &G::hCoef_HardnessMin,sizeof(float),cudaMemcpyHostToDevice);	
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_SoftenRatePtr, (void const *) &G::hCoef_SoftenRate,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_ThermalErosionRatePtr, (void const *) &G::hCoef_ThermalErosionRate,sizeof(float),cudaMemcpyHostToDevice);
	G::cuda_ERR = cudaMemcpy((void *) G::hCoef_EvaporationRatePtr, (void const *) &G::hCoef_EvaporationRate,sizeof(float),cudaMemcpyHostToDevice);

	//Initialization of Arrays(Buffers)
	
	//ptrArray - points to the currently modified array
	float *ptrArray = cuda_fetchTerrainHptr();
	//Initialize Terrain Heights
	G::cuda_ERR = cudaMemcpy2D(ptrArray, 2048, &G::initialTerrainH[0], 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyHostToDevice);	
	G::cuda_ERR = cudaGetLastError();

	//Initialize Water Heights
	ptrArray = cuda_fetchWaterHptr();	
	G::cuda_ERR = cudaMemcpy2D(ptrArray, 2048, &G::initialHardness[0], 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyHostToDevice);	
	G::cuda_ERR = cudaGetLastError();

	ptrArray = cuda_fetchHardnessptr();
	//Initialize Hardness
	G::cuda_ERR = cudaMemcpy2D(ptrArray, 2048, &G::initialHardness[0], 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyHostToDevice);	
	G::cuda_ERR = cudaGetLastError();
	
	ptrArray = cuda_fetchRainRateptr();
	//Initialize RainRate
	G::cuda_ERR = cudaMemcpy2D(ptrArray, 2048, &G::initialRainRate[0], 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyHostToDevice);	
	G::cuda_ERR = cudaGetLastError();

	//cuda_SetRainRate((void const *)&G::initialHardness[0], (size_t) G::gridSideCnt*sizeof(float),G::gridSideCnt*sizeof(float), G::gridSideCnt);
	//G::cuda_ERR = cudaMemset2D(dWaterRainRate0_ptr,G::gridSideCnt*sizeof(float),0,G::gridSideCnt*sizeof(float),G::gridSideCnt);
	//int test = cuda_SetTerrainHeight(&G::initialTerrainH[0], (size_t) G::gridSideCnt*sizeof(float),G::gridSideCnt*sizeof(float), G::gridSideCnt);
	
	//cuda_SetHardness((void const **)&G::initialHardness[0], (size_t) G::gridSideCnt*sizeof(float),G::gridSideCnt*sizeof(float), G::gridSideCnt);
	
	
	
	//cudaGetSymbolSize(&sz,"dTerrainH0_ptr");	

	//memset(&G::initialTerrainH[0],0,512*512*4);
	//G::cuda_ERR=cudaMemcpy2D(&G::initialTerrainH[0], 2048,ad , 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyDeviceToHost);	
	//cudaMemcpyToSymbol("dTerrainH0_ptr",&G::initialTerrainH[0],512*512*4,0,cudaMemcpyHostToDevice);
	//memset(&G::initialTerrainH[0],0,512*512*4);
	//G::cuda_ERR=cudaMemcpy2D(&G::initialTerrainH[0], 2048,ad , 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyDeviceToHost);	
}


void initGL()
{
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	glEnable( GL_TEXTURE_2D );
	glEnable( GL_DEPTH_TEST );	
	glDepthFunc( GL_LEQUAL );

    glClearColor( 0.0, 0.0, 0.0, 1.0 ); 

    int NumVerticesSide = 512;
	G::gridSideCnt = NumVerticesSide;
	G::gridVertexCnt = NumVerticesSide*NumVerticesSide;
  	G::gridTriCnt = (NumVerticesSide-1)*(NumVerticesSide-1)*6;
	
	generate_vectorField_mesh(&G::gridVectorPositions[0],NumVerticesSide,NumVerticesSide,1.0);
	generate_heightfield_mesh(&G::gridMeshPositions[0],&G::gridMeshIndices[0], NumVerticesSide, NumVerticesSide, 1.0);
	generate_heightfield_heights(&G::gridMeshPositions[0],&G::initialTerrainH[0],NumVerticesSide, NumVerticesSide, 8, 25.0, 1.0);
	generate_heightfield_hardness(&G::initialHardness[0],NumVerticesSide,NumVerticesSide,3,0.0,.98);
	generate_heightfield_rain(&G::initialRainRate[0],NumVerticesSide,NumVerticesSide,3,0.1,2.0);

	int sizeOfPoints = G::gridVertexCnt*sizeof(vec4);

    // Create a vertex array object
    GLuint vao;	
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );
	
	
    glGenBuffers( 1, &buffer );
    glBindBuffer( GL_ARRAY_BUFFER, buffer );
    glBufferData( GL_ARRAY_BUFFER, sizeOfPoints, G::gridMeshPositions, GL_DYNAMIC_DRAW );	
	GLuint test=(int)&G::gridMeshIndices[0];
	GLuint test2=test+G::gridTriCnt*3*sizeof(GLuint);

	
	glGenBuffers(1, &indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, G::gridTriCnt*sizeof(GLuint), G::gridMeshIndices, GL_STATIC_DRAW);

	//ground heightmap
	int num_texels = G::gridSideCnt * G::gridSideCnt;
    int texels_size = sizeof(vec4) * num_texels;
	glGenBuffers(1,&G::pboGroundH);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, G::pboGroundH);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, texels_size, NULL, GL_DYNAMIC_COPY); 

	for(int i=0;i<G::gridSideCnt;i++){for(int j=0;j<G::gridSideCnt;j++){
		tester[i*G::gridSideCnt+j].x=(rand());
		tester[i*G::gridSideCnt+j].y=(rand());
		tester[i*G::gridSideCnt+j].z=(rand());
		tester[i*G::gridSideCnt+j].w=(rand());
	}}
	glBufferSubData(GL_PIXEL_UNPACK_BUFFER,0,texels_size,tester);
	
	glGenTextures(1,&G::texGroundH);  	
	glBindTexture( GL_TEXTURE_2D, G::texGroundH); 		
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, G::gridSideCnt, G::gridSideCnt, 0, GL_RGBA, GL_FLOAT, BUFFER_OFFSET(0));	 
	
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f );

    // Load shaders and use the resulting shader program
	G::shaderGround = InitShader( "vshader42.glsl", "fshader42.glsl" );
	glUseProgram( G::shaderGround );

    // set up vertex arrays
	G::shaderVPosition = glGetAttribLocation( G::shaderGround, "vPosition" );
	glEnableVertexAttribArray( G::shaderVPosition );
	glVertexAttribPointer( G::shaderVPosition, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );
    G::shaderModel_view = glGetUniformLocation( G::shaderGround, "model_view" );
	G::shaderMousePosition = glGetUniformLocation(G::shaderGround, "mouse" );
	G::shaderWorldScale = glGetUniformLocation(G::shaderGround,"worldScale" );
	G::shaderProjection = glGetUniformLocation( G::shaderGround, "projection" );	
	G::shaderEditDistance =glGetUniformLocation( G::shaderGround, "editDistance");
	G::tex_loc = glGetUniformLocation(G::shaderGround, "heights");	
	G::vShader_tex_loc = glGetUniformLocation(G::shaderGround, "heightsV");
	G::type = glGetUniformLocation(G::shaderGround, "type");
}


void display( void )
{
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	//Copy PBO to texture
	glBindTexture(GL_TEXTURE_2D, G::texGroundH);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, G::pboGroundH);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G::gridSideCnt, G::gridSideCnt, GL_RGBA, GL_FLOAT, BUFFER_OFFSET(0));
	glUniform1i(G::tex_loc, 0);
	glUniform1i(G::vShader_tex_loc, 0);
	
	
    vec4    up( 0.0, 1.0, 0.0, 0.0 );
	
	
	G::mv = LookAt( G::uiCamera.pos, G::uiCamera.pos+G::uiCamera.dir*4.0, up )*Angel::Scale(G::hCoef_PipeLength,1.0,G::hCoef_PipeLength);
	glUniformMatrix4fv( G::shaderModel_view, 1, GL_TRUE, G::mv );

	glUniform2f(G::shaderMousePosition,G::uimouseWorld.x,G::uimouseWorld.z);
	glUniform1f(G::shaderWorldScale,G::hCoef_PipeLength);
	glUniform1f(G::shaderEditDistance,G::uiEditRadius);
    
	G::p = Perspective( G::uiFovy, G::uiAspect,G::uizNear, G::uizFar);
	glUniformMatrix4fv( G::shaderProjection, 1, GL_TRUE, G::p );	
	//glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

	
	//Draw vector field
	/*glBufferData( GL_ARRAY_BUFFER, sizeof(vec4)*G::gridVertexCnt*2, G::gridVectorPositions, GL_STATIC_DRAW );
	glUniformMatrix4fv( G::shaderModel_view, 1, GL_TRUE, G::mv );
	glUniform1i(G::type, 3);
	glDrawArrays(GL_LINES,0,G::gridVertexCnt*2);*/
	
	//End of vector field ops

	//Bind Triangle Data
	//glBufferData( GL_ARRAY_BUFFER, sizeof(vec4)*G::gridVertexCnt, G::gridMeshPositions, GL_STATIC_DRAW );

	//DELETE THIS SHIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//TEEEEEEEEST
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	mat4 testMV = LookAt( G::uiCamera.pos, G::uiCamera.pos+G::uiCamera.dir*4.0, up )*Angel::Translate(G::uimouseWorld.x,G::uimouseWorld.y,G::uimouseWorld.z)*Angel::Scale(10.,1.0,10.)*Angel::Translate(0.0,4.,0.0)*Angel::Scale(1.,0.0,1.)*Angel::Translate(255.5*G::hCoef_PipeLength,2.,-255.5*G::hCoef_PipeLength);
	glUniformMatrix4fv( G::shaderModel_view, 1, GL_TRUE, testMV );
	glUniform1i(G::type, 0);
	glDrawElements(GL_TRIANGLES, 8, GL_UNSIGNED_INT,BUFFER_OFFSET(0));	
	glUniformMatrix4fv( G::shaderModel_view, 1, GL_TRUE, G::mv );
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	//DEEEEEEEEEEEEEEEELEEEEEEEEEEEEEEEEEEEEEEETEEEEEEEEEEEEEE
	
	glUniform1i(G::type, 0);
	glDrawElements(GL_TRIANGLES, G::gridTriCnt, GL_UNSIGNED_INT,BUFFER_OFFSET(0));
	
	//glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	
	//Depth Prepass
	glUniform1i(G::type, 2);
	glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
	glDrawElements(GL_TRIANGLES, G::gridTriCnt, GL_UNSIGNED_INT,BUFFER_OFFSET(0));		
	glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
	
	glUniform1i(G::type, 1);
	glDepthMask(GL_FALSE);
	glDrawElements(GL_TRIANGLES, G::gridTriCnt, GL_UNSIGNED_INT,BUFFER_OFFSET(0));
	glDepthMask(GL_TRUE);
	
	//glPointSize(5.0);
	//glDrawElements(GL_POINTS, G::gridTriCnt,GL_UNSIGNED_INT,BUFFER_OFFSET(0));

	
	
    glutSwapBuffers();

}


void UpdateCamera(Camera *camera)
{
	
	float mouseDx = G::uiMouseNewX - G::uiMouseOldX;
	float mouseDy = G::uiMouseNewY - G::uiMouseOldY;
	//printf("\n\t %f %f",mouseDx, mouseDy);
	G::uiMouseOldY = G::uiMouseNewY;
	G::uiMouseOldX = G::uiMouseNewX;
	if (G::uiMouse_M)
	{
		camera->phi = camera->phi + mouseDx * C::CAM_ROT_FACTOR_X * DegreesToRadians;
		camera->theta = camera->theta - mouseDy * C::CAM_ROT_FACTOR_Y * DegreesToRadians;

		//Wrap around
		if(camera->phi > 2.0 * M_PI)camera->phi -= 2.0 * M_PI;
		else if(camera->phi < -2.0 * M_PI)camera->phi += 2.0 * M_PI;
				
		//Set vertical limits
		if(camera->theta > M_PI * 0.5)camera->theta = M_PI * 0.5;
		else if(camera->theta < M_PI * -0.5 + 0.15)camera->theta = M_PI * -0.5 + 0.15;

		//printf("\n %f %f",camera->phi,camera->theta);
		GLfloat phi = camera->phi;
		GLfloat theta = camera->theta;

		camera->dir = normalize(point4( cos(phi)*cos(theta),
								   sin(theta),
								   sin(phi)*cos(theta),
								   0.0
								  ));
		camera->dir[3] = 0.0;
		camera->side = normalize(cross(camera->dir,vec4(0.0,1.0,0.0,0.0)));
	}
	//Moving Camera
	if (G::uiKey_W)
	{
		camera->pos = camera->pos + camera->dir * C::CAM_MOV_SPEED * G::dt;
		camera->pos[3] = 1.0;
	}
	if (G::uiKey_S)
	{
		camera->pos = camera->pos - camera->dir * C::CAM_MOV_SPEED * G::dt;
		camera->pos[3] = 1.0;
	}
	if (G::uiKey_A)
	{
		camera->pos = camera->pos - camera->side * C::CAM_MOV_SPEED * G::dt;
		camera->pos[3] = 1.0;
	}
	if (G::uiKey_D)
	{
		camera->pos = camera->pos + camera->side * C::CAM_MOV_SPEED * G::dt;
		camera->pos[3] = 1.0;
	}
	if (G::uiKey_R)
	{
		camera->pos = vec4(0.0,78.0,0.0,1.0);
		camera->dir = vec4(0.5,-11.0,0.0,0.0);
		camera->phi = 0.0;
		camera->theta = -M_PI*0.5;
	}
	
}

void unproject(float x, float y, float z)
{
	vec4    up( 0.0, 1.0, 0.0, 0.0 );
	int mViewport[4]={0,0,G::uiwinX,G::uiwinY};
	GLdouble mView[16]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	GLdouble mProjection[16]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++)
		{
			mView[i*4+j]=G::mv[j][i];
			mProjection[i*4+j]=G::p[j][i];
		}
	}
	double worldX,worldY,worldZ;
	gluUnProject(GLdouble(x),GLdouble(y),GLdouble(z),mView,mProjection,mViewport,&worldX,&worldY,&worldZ);
	G::uimouseWorld.x=(float)worldX;
	G::uimouseWorld.y=(float)worldY;
	G::uimouseWorld.z=(float)worldZ;
}
int mode=0;
int modeSentinel=0;
int walk=0;
int walkSentinel=0;
void HandleInput()
{
	float z;
	glReadPixels(G::uiMouseNewX,G::uiwinY-G::uiMouseNewY,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&z);
	unproject(G::uiMouseNewX,G::uiwinY-G::uiMouseNewY,z);

	UpdateCamera(&G::uiCamera);		
	if(G::uiKey_Q&&!modeSentinel)
	{
		//printf("q");
		mode^=1;
		modeSentinel=1;
	}
	if(G::uiKey_Space&&!walkSentinel)
	{
		walk^=1;
		walkSentinel = 1;
	}
	printf("\n\t\t@%f %f %f",G::uimouseWorld.x,G::uimouseWorld.z,G::uimouseWorld.y);
	if (G::uiMouse_L||G::uiMouse_R)
	{		
		CollisionTest = G::uimouseWorld;
		float sign;		
		if (G::uiMouse_L) sign=1.0; 
		else sign = -1.0;

		//printf("\n\t\tEDIT @%f %f %f",G::uimouseWorld.x,G::uimouseWorld.z,G::uimouseWorld.y);
		cuda_EditTerrain(G::uimouseWorld.x*G::hCoef_PipeLength, G::uimouseWorld.z*G::hCoef_PipeLength, G::hCoef_PipeLength, 25.0*sign, 512, 512 ,G::uiEditRadius, G::dt, mode );
	}
}
//Called On Every Frame
float heightTest[512*512];
vec2 vectorTest[512*512];
void Tick(int value)
{
	G::iterations++;
	//Schedule a new tick, prevents more than 60 FPS (max display refresh rate)
	glutTimerFunc(16, Tick, 0);	
	//system("CLS");
	//printf("\n\n---------Tick %i-----------Mode %i", G::iterations, mode);
	frustumPlanes();

	HandleInput();
		
	runCUDA();	
	//Update the display
	glutPostRedisplay();
	
	//Debug Output
	//float *ptrArray = cuda_fetchSedimentAmountPtr();
	float *ptrArray = cuda_fetchTerrainHptr();
	G::cuda_ERR = cudaMemcpy2D(heightTest, 2048, ptrArray, 2048, G::gridSideCnt*sizeof(float), G::gridSideCnt, cudaMemcpyDeviceToHost);	
	G::cuda_ERR = cudaGetLastError();
	//ptrArray = cuda_fetchWaterVelocityPtr();
	//G::cuda_ERR = cudaMemcpy2D((void *)vectorTest, 2048*2, ptrArray, 2048*2, G::gridSideCnt*sizeof(float)*2, G::gridSideCnt, cudaMemcpyDeviceToHost);	
	//G::cuda_ERR = cudaGetLastError();

	float heightSum=0.0;
	for(int i=0;i<512*512;i++){heightSum+=heightTest[i];}
	printf("\n\n HeightSum %f\n\n", heightSum);
	////Data Probe. Displays data around cursor using the console
	int x = fminf(fmaxf((G::uimouseWorld.x+256.0)/G::hCoef_PipeLength,5),507);
	int y = fminf(fmaxf((G::uimouseWorld.z+256.0)/G::hCoef_PipeLength,5),507);
	printf("\n     x:%i y:%i \n",x,y);
	for(int i=y-3;i<y+2;i++)
	{
		for(int j=x-3;j<x+2;j++)
		{
			if(i==y-1&&j==x-1)
				printf("[%6.2f]", heightTest[i*512+j]);
			else 
				printf(" %6.2f ", heightTest[i*512+j]);
		}
		printf("\n\n");
	}
	
	//Walk on terrain
	if(walk)
	G::uiCamera.pos.y = heightTest[(int((G::uiCamera.pos.z+256.0)/G::hCoef_PipeLength))*512 + int((G::uiCamera.pos.x+256.0)/G::hCoef_PipeLength)]+2.0;
	
	//Timing
	G::time_Previous=G::time_Current;
	G::time_Current=glutGet(GLUT_ELAPSED_TIME);
	G::dt=float(G::time_Current-G::time_Previous)/1000.0;
		
	sprintf(G::uiwinTitle,"\n %f sec",G::dt);
	glutSetWindowTitle(G::uiwinTitle);
	//printf("\n\n---------END-----------");
	
}

void mousePressed(int button, int state, int x, int y)
{
	int pressed = 0;
	if(GLUT_DOWN==state) pressed = 1;
	
	if (GLUT_LEFT_BUTTON==button) G::uiMouse_L = pressed;
	else if (GLUT_RIGHT_BUTTON==button) G::uiMouse_R = pressed;
	else if (GLUT_MIDDLE_BUTTON==button)G::uiMouse_M = pressed;
	
	G::uiMouseOldX = G::uiMouseNewX = x;
	G::uiMouseOldY = G::uiMouseNewY = y;
	//G::uiMouseNewX = x;
	//G::uiMouseNewY = y;
}
void mouseWheeled(int wheel, int dir, int x, int y)
{
	G::uiMouseOldX = G::uiMouseNewX;
	G::uiMouseOldY = G::uiMouseNewY;
	G::uiMouseNewX = x;
	G::uiMouseNewY = y;
	G::uiEditRadius+=0.5*dir;
}

void mouseMotion(int x, int y)
{
	G::uiMouseOldX = G::uiMouseNewX;
	G::uiMouseOldY = G::uiMouseNewY;
	G::uiMouseNewX = x;
	G::uiMouseNewY = y;
}
void mousePassiveMotion(int x, int y)
{
	//No mouse button pressed, reset the mouse buttons' state
	G::uiMouse_L = 0;
	G::uiMouse_R = 0;
	G::uiMouse_M = 0;

	G::uiMouseOldX = G::uiMouseNewX;
	G::uiMouseOldY = G::uiMouseNewY;
	G::uiMouseNewX = x;
	G::uiMouseNewY = y;
}

void keyboard( unsigned char key, int x, int y )
{
    if(033==key) exit( EXIT_SUCCESS );
	else if('w'==key) G::uiKey_W = 1;
	else if('s'==key) G::uiKey_S = 1;
	else if('a'==key) G::uiKey_A = 1;
	else if('d'==key) G::uiKey_D = 1;

	else if('q'==key) G::uiKey_Q = 1;
	else if('e'==key) G::uiKey_E = 1;
	else if('r'==key) G::uiKey_R = 1;
	else if('p'==key) G::uiKey_P = 1;
	else if(' '==key) G::uiKey_Space =1;
}

void keyboardUp( unsigned char key, int x, int y )
{
	if('w'==key) G::uiKey_W = 0;
	else if('s'==key) G::uiKey_S = 0;
	else if('a'==key) G::uiKey_A = 0;
	else if('d'==key) G::uiKey_D = 0;
	
	else if('q'==key) {G::uiKey_Q = 0;modeSentinel=0;}
	else if('e'==key) G::uiKey_E = 0;
	else if('r'==key) G::uiKey_R = 0;
	else if('p'==key) G::uiKey_P = 0;
	else if(' '==key) {G::uiKey_Space =0;walkSentinel=0;}
}

void reshapeWindow( int width, int height )
{
    glViewport( 0, 0, width, height );
	G::uiwinX = width;
	G::uiwinY = height;
	G::uiAspect = GLfloat(width)/height;
}

void initGlutCallbacks()
{
	//Render
	glutDisplayFunc( display );
	
	//Window Resize
	glutReshapeFunc( reshapeWindow );
	
	//Various Input
    glutKeyboardFunc( keyboard );
	glutKeyboardUpFunc(keyboardUp);
	glutIgnoreKeyRepeat(1);
	
	glutMouseFunc(mousePressed);
	glutMouseWheelFunc(mouseWheeled);
	glutMotionFunc(mouseMotion);
	glutPassiveMotionFunc(mousePassiveMotion);    	
}


int main( int argc, char **argv )
{
	G::cuda_ERR = cudaGLSetGLDevice(0);
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
    glutInitWindowSize( 512, 512 );
    glutInitContextVersion( 3, 2 );
    glutInitContextProfile( GLUT_CORE_PROFILE );
    glutCreateWindow( "Terrain Generator" );
	glewExperimental = GL_TRUE;
    glewInit();
	
    initGL();
	initCUDA();
	initGlutCallbacks();
    
	G::uiCamera.pos = vec4(0.0,70.0,0.0,1.0);
	G::uiCamera.dir = vec4(0.5,-11.0,0.0,0.0);
	G::uiCamera.phi = 0.0;
	G::uiCamera.theta = -M_PI*0.5;

	glutTimerFunc(16, Tick, 0);
    glutMainLoop();
    return 0;
}

