
//The correct one


#include <stdio.h>
#include "cutil_math.h"
//TICK VERY IMPORTANT, used for swapping the ping pong
int tick=0;
//All cuda stuff goes here

//Global Data Pointers

//Water
__device__ float *dWaterRainRate0_ptr; 
__device__ float *dWaterRainRate1_ptr; 
__device__ float *dWaterH0_ptr;		   
__device__ float *dWaterH1_ptr;		   
__device__ float4 *dWaterFlux0_ptr;      
__device__ float4 *dWaterFlux1_ptr;
__device__ float2 *dWaterVelocity_ptr;    

//Sediment
__device__ float *dSedimentCapacity_ptr; 
__device__ float *dSedimentAmount0_ptr;	 
__device__ float *dSedimentAmount1_ptr;	 
__device__ float *dSedimentAmntAdvect_ptr;
__device__ float *dSedimentAmntAdvectBack_ptr; 


//Terrain
__device__ float *dTerrainH0_ptr;
__device__ float *dTerrainH1_ptr;	   
__device__ float *dHardness0_ptr;	   
__device__ float *dHardness1_ptr;	   

//Thermal Erosion
__device__ float *dThermalAmount2Move_ptr;
__device__ float4 *dThermalFlux_ptr;		 
__device__ float4 *dThermalFluxDiag_ptr;


//Pitches STORED ON HOST!!!!!!
size_t hWaterRainRate0_pitch=0;
size_t hWaterRainRate1_pitch=0;
size_t hWaterH0_pitch=0;		
size_t hWaterH1_pitch=0;		
size_t hWaterFlux_pitch=0;     
size_t hWaterVelocity_pitch=0;   

//Sediment
size_t hSedimentCapacity_pitch=0;
size_t hSedimentAmount_pitch=0;

//Terrain
size_t hTerrainH0_pitch=0;	
size_t hTerrainH1_pitch=0;	
size_t hHardness0_pitch=0;	
size_t hHardness1_pitch=0;	

//Thermal Erosion
size_t hThermalAmount2Move_pitch=0;
size_t hThermalFlux_pitch=0;
size_t hThermalFluxDiag_pitch=0;

//Textures
//For each: texture, pointer to it, offset
texture<float, cudaTextureType2D, cudaReadModeElementType> dTexWaterRainRate; 
textureReference const * dTexWaterRainRatePtr;
size_t dTexWaterRainRateOffset;
texture<float, cudaTextureType2D, cudaReadModeElementType> dTexWaterH; 
textureReference const *dTexWaterHptr;
size_t dTexWaterHOffset;
texture<float4, cudaTextureType2D, cudaReadModeElementType> dTexWaterFlux; 
textureReference const *dTexWaterFluxPtr;
size_t dTexWaterFluxOffset;
texture<float2, cudaTextureType2D, cudaReadModeElementType> dTexWaterVelocity; 
textureReference const *dTexWaterVelocityPtr;
size_t dTexWaterVelocityOffset;

texture<float, cudaTextureType2D, cudaReadModeElementType> dTexSedimentCapacity; 
textureReference const *dTexSedimentCapacityPtr;
size_t dTexSedimentCapacityOffset;
texture<float, cudaTextureType2D, cudaReadModeElementType> dTexSedimentAmount; 
textureReference const *dTexSedimentAmountPtr;
size_t dTexSedimentAmountOffset;
texture<float, cudaTextureType2D, cudaReadModeElementType> dTexSedimentAmntAdvect; 
textureReference const *dTexSedimentAmntAdvectPtr;
size_t dTexSedimentAmntAdvectOffset;
texture<float, cudaTextureType2D, cudaReadModeElementType> dTexSedimentAmntAdvectBack; 
textureReference const *dTexSedimentAmntAdvectBackPtr;
size_t dTexSedimentAmntAdvectBackOffset;

texture<float, cudaTextureType2D, cudaReadModeElementType> dTexTerrainH; 
textureReference const *dTexTerrainHptr;
size_t dTexTerrainHOffset;
texture<float, cudaTextureType2D, cudaReadModeElementType> dTexHardness; 
textureReference const *dTexHardnessPtr;
size_t dTexHardnessOffset;

texture<float, cudaTextureType2D, cudaReadModeElementType> dTexThermalAmnt2Move; 
textureReference const *dTexThermalAmnt2MovePtr;
size_t dTexThermalAmnt2MoveOffset;
texture<float4, cudaTextureType2D, cudaReadModeElementType> dTexThermalFlux; 
textureReference const *dTexThermalFluxPtr;
size_t dTexThermalFluxOffset;
texture<float4, cudaTextureType2D, cudaReadModeElementType> dTexThermalFluxDiag; 
textureReference const *dTexThermalFluxDiagPtr;
size_t dTexThermalFluxDiagOffset;
//Texture Descriptors

struct cudaChannelFormatDesc texFloatChannelDesc;
struct cudaChannelFormatDesc texFloat2ChannelDesc;
struct cudaChannelFormatDesc texFloat4ChannelDesc;
//------------------------------------------------------------------------------------

//Constants. Defaults chosen from Balazs Jako's Paper
__constant__ float dCoef_Timestep = 0.02f;
__constant__ float dCoef_G = 9.81f;
__constant__ float dCoef_PipeCrossSection = 40.0f;
__constant__ float dCoef_PipeLength = 1.0f ;

__constant__ float dCoef_RainRate = 0.012f;

__constant__ float dCoef_talusRatio = 1.2f;
__constant__ float dCoef_talusCoef = 0.8f;
__constant__ float dCoef_talusBias = 0.1f;

__constant__ float dCoef_SedimentCapacityFactor = 1.0f;
__constant__ float dCoef_DepthMax = 10.0f;

__constant__ float dCoef_DissolveRate = 0.5f;
__constant__ float dCoef_SedimentDropRate = 1.0f;
__constant__ float dCoef_HardnessMin = 0.5f;
__constant__ float dCoef_SoftenRate = 5.0f;

__constant__ float dCoef_ThermalErosionRate = 0.15f;

__constant__ float dCoef_EvaporationRate = 0.015f;

extern "C" void cuda_Initialize(int width, int height)
{
	printf("\n\n\n %i %i", width, height);
	texFloatChannelDesc = cudaCreateChannelDesc<float1>();
	texFloat2ChannelDesc = cudaCreateChannelDesc<float2>();
	texFloat4ChannelDesc = cudaCreateChannelDesc<float4>();

	dTexWaterRainRate.normalized = true;
	dTexWaterRainRate.filterMode = cudaFilterModeLinear;
	dTexWaterRainRate.addressMode[0] = cudaAddressModeWrap;
	dTexWaterRainRate.addressMode[1] = cudaAddressModeWrap;
	
	dTexWaterH.normalized = true;
	dTexWaterH.filterMode = cudaFilterModeLinear;
	dTexWaterH.addressMode[0] = cudaAddressModeWrap;
	dTexWaterH.addressMode[1] = cudaAddressModeWrap;

	dTexWaterFlux.normalized = true;
	dTexWaterFlux.filterMode = cudaFilterModeLinear;
	dTexWaterFlux.addressMode[0] = cudaAddressModeWrap;
	dTexWaterFlux.addressMode[1] = cudaAddressModeWrap;

	dTexWaterVelocity.normalized = true;
	dTexWaterVelocity.filterMode = cudaFilterModeLinear;
	dTexWaterVelocity.addressMode[0] = cudaAddressModeWrap;
	dTexWaterVelocity.addressMode[1] = cudaAddressModeWrap;

	dTexSedimentCapacity.normalized = true;
	dTexSedimentCapacity.filterMode = cudaFilterModeLinear;
	dTexSedimentCapacity.addressMode[0] = cudaAddressModeWrap;
	dTexSedimentCapacity.addressMode[1] = cudaAddressModeWrap;

	
	
	dTexSedimentAmount.normalized = true;
	dTexSedimentAmount.filterMode = cudaFilterModeLinear;
	dTexSedimentAmount.addressMode[0] = cudaAddressModeWrap;
	dTexSedimentAmount.addressMode[1] = cudaAddressModeWrap;

	dTexSedimentAmntAdvect.normalized = true;
	dTexSedimentAmntAdvect.filterMode = cudaFilterModeLinear;
	dTexSedimentAmntAdvect.addressMode[0] = cudaAddressModeWrap;
	dTexSedimentAmntAdvect.addressMode[1] = cudaAddressModeWrap;

	dTexSedimentAmntAdvectBack.normalized = true;
	dTexSedimentAmntAdvectBack.filterMode = cudaFilterModeLinear;
	dTexSedimentAmntAdvectBack.addressMode[0] = cudaAddressModeWrap;
	dTexSedimentAmntAdvectBack.addressMode[1] = cudaAddressModeWrap;
	


	dTexTerrainH.normalized = true;
	dTexTerrainH.filterMode = cudaFilterModeLinear;
	dTexTerrainH.addressMode[0] = cudaAddressModeWrap;
	dTexTerrainH.addressMode[1] = cudaAddressModeWrap;

	dTexHardness.normalized = true;
	dTexHardness.filterMode = cudaFilterModeLinear;
	dTexHardness.addressMode[0] = cudaAddressModeWrap;
	dTexHardness.addressMode[1] = cudaAddressModeWrap;

	dTexThermalAmnt2Move.normalized = true;
	dTexThermalAmnt2Move.filterMode = cudaFilterModeLinear;
	dTexThermalAmnt2Move.addressMode[0] = cudaAddressModeWrap;
	dTexThermalAmnt2Move.addressMode[1] = cudaAddressModeWrap;

	dTexThermalFlux.normalized = true;
	dTexThermalFlux.filterMode = cudaFilterModeLinear;
	dTexThermalFlux.addressMode[0] = cudaAddressModeWrap;
	dTexThermalFlux.addressMode[1] = cudaAddressModeWrap;

	dTexThermalFluxDiag.normalized = true;
	dTexThermalFluxDiag.filterMode = cudaFilterModeLinear;
	dTexThermalFluxDiag.addressMode[0] = cudaAddressModeWrap;
	dTexThermalFluxDiag.addressMode[1] = cudaAddressModeWrap;

	cudaGetTextureReference(&dTexWaterRainRatePtr, "dTexWaterRainRate");
	cudaGetTextureReference(&dTexWaterHptr, "dTexWaterH");
	cudaGetTextureReference(&dTexWaterFluxPtr, "dTexWaterFlux");
	cudaGetTextureReference(&dTexWaterVelocityPtr, "dTexWaterVelocity");
	cudaGetTextureReference(&dTexSedimentCapacityPtr, "dTexSedimentCapacity");
	cudaGetTextureReference(&dTexSedimentAmountPtr, "dTexSedimentAmount");
	cudaGetTextureReference(&dTexSedimentAmntAdvectPtr, "dTexSedimentAmntAdvect");
	cudaGetTextureReference(&dTexSedimentAmntAdvectBackPtr, "dTexSedimentAmntAdvectBack");
	cudaGetTextureReference(&dTexTerrainHptr, "dTexTerrainH");
	cudaGetTextureReference(&dTexHardnessPtr, "dTexHardness");
	cudaGetTextureReference(&dTexThermalAmnt2MovePtr, "dTexThermalAmnt2Move");
	cudaGetTextureReference(&dTexThermalFluxPtr, "dTexThermalFlux");
	cudaGetTextureReference(&dTexThermalFluxDiagPtr, "dTexThermalFluxDiag");

//Allocate Device Memory	
	cudaMallocPitch((void**)&dWaterRainRate0_ptr,&hWaterRainRate0_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dWaterRainRate1_ptr,&hWaterRainRate1_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dWaterH0_ptr,&hWaterH0_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dWaterH1_ptr,&hWaterH1_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dWaterFlux0_ptr,&hWaterFlux_pitch,(width*sizeof(float4)),height);
	cudaMallocPitch((void**)&dWaterFlux1_ptr,&hWaterFlux_pitch,(width*sizeof(float4)),height);
	cudaMallocPitch((void**)&dWaterVelocity_ptr,&hWaterVelocity_pitch,(width*sizeof(float2)),height);
	printf("\n\n\n %i %i", width, height);
	cudaMallocPitch((void**)&dSedimentCapacity_ptr,&hSedimentCapacity_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dSedimentAmount0_ptr,&hSedimentAmount_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dSedimentAmount1_ptr,&hSedimentAmount_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dSedimentAmntAdvect_ptr,&hSedimentAmount_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dSedimentAmntAdvectBack_ptr,&hSedimentAmount_pitch,(width*sizeof(float)),height);

	cudaMallocPitch((void**)&dTerrainH0_ptr,&hTerrainH0_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dTerrainH1_ptr,&hTerrainH1_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dHardness0_ptr,&hHardness0_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dHardness1_ptr,&hHardness1_pitch,(width*sizeof(float)),height);

	cudaMallocPitch((void**)&dThermalAmount2Move_ptr,&hThermalAmount2Move_pitch,(width*sizeof(float)),height);
	cudaMallocPitch((void**)&dThermalFlux_ptr,&hThermalFlux_pitch,(width*sizeof(float4)),height);
	cudaMallocPitch((void**)&dThermalFluxDiag_ptr,&hThermalFluxDiag_pitch,(width*sizeof(float4)),height);

	//Memset Device Memory
	cudaMemset2D(dWaterRainRate0_ptr,hWaterRainRate0_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dWaterRainRate1_ptr,hWaterRainRate1_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dWaterH0_ptr,hWaterH0_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dWaterH1_ptr,hWaterH1_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dWaterFlux0_ptr,hWaterFlux_pitch,0,width*sizeof(float4),height);
	cudaMemset2D(dWaterFlux1_ptr,hWaterFlux_pitch,0,width*sizeof(float4),height);
	cudaMemset2D(dWaterVelocity_ptr,hWaterVelocity_pitch,0,width*sizeof(float2),height);
	cudaMemset2D(dSedimentCapacity_ptr,hSedimentCapacity_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dSedimentAmount0_ptr,hSedimentAmount_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dSedimentAmount1_ptr,hSedimentAmount_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dSedimentAmntAdvect_ptr,hSedimentAmount_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dSedimentAmntAdvectBack_ptr,hSedimentAmount_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dTerrainH0_ptr,hTerrainH0_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dTerrainH1_ptr,hTerrainH1_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dHardness0_ptr,hHardness0_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dHardness1_ptr,hHardness1_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dThermalAmount2Move_ptr,hThermalAmount2Move_pitch,0,width*sizeof(float),height);
	cudaMemset2D(dThermalFlux_ptr,hThermalFlux_pitch,0,width*sizeof(float4),height);
	cudaMemset2D(dThermalFluxDiag_ptr,hThermalFluxDiag_pitch,0,width*sizeof(float4),height);
}

//USELESS
extern "C" int cuda_SetTerrainHeight(void * src, size_t srcPitch,size_t width, size_t height)
{
	return cudaMemcpy2D(dTerrainH0_ptr, hTerrainH0_pitch, src, srcPitch, width, height, cudaMemcpyHostToDevice);
}
//USELESS
extern "C" int cuda_SetHardness(void * src, size_t srcPitch,size_t width, size_t height)
{
	return cudaMemcpy2D(dHardness0_ptr, hHardness0_pitch, src, srcPitch, width, height, cudaMemcpyHostToDevice);
	
}
//USELESS
extern "C" int cuda_SetRainRate(void * src, size_t srcPitch,size_t width, size_t height)
{
	return cudaMemcpy2D(dWaterRainRate0_ptr, hWaterRainRate0_pitch, src, srcPitch, width, height, cudaMemcpyHostToDevice);	
}

extern "C" float* cuda_fetchTerrainHptr()
{
	return dTerrainH0_ptr;
}
extern "C" float* cuda_fetchWaterHptr()
{
	return dWaterH0_ptr;
}
extern "C" float* cuda_fetchRainRateptr()
{
	return dWaterRainRate0_ptr;
}
extern "C" float* cuda_fetchHardnessptr()
{
	return dHardness0_ptr;
}

extern "C" float* cuda_fetchWaterVelocityPtr()
{
	return (float*)dWaterVelocity_ptr;
}

extern "C" float* cuda_fetchSedimentAmountPtr()
{
	return (float*)dSedimentAmount0_ptr;
}



//Testing Kernel
__global__ void kernelTest(float4 *heights, size_t width, unsigned int height, float dt, float*in, float*in2)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float a = tex2D(dTexTerrainH,(x+0.5f)/height,(y+.5f)/height);
	float b = tex2D(dTexWaterRainRate,(x+0.5f)/height,(y+0.5f)/height);
	heights[y*width+x].x=sinf(x*0.1f+dt)*cosf(y*0.1f+dt);	
	heights[y*width+x].y = b;
	heights[y*width+x].z = a;
	heights[y*width+x].w=cosf(y*0.1f+dt)*10.0;

	in[y*width+x] = a;
	in2[y*width+x] = b;
}

__global__ void kernelEditBuffer(float* in, float * out, float editX, float editY, float pointDistance, float editValue, size_t width, unsigned int 
height ,float maxDist, float dt )
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	float2 pos;
	pos.x = ((float)x-0.5f*height+0.5f)*pointDistance;
	pos.y = ((float)y-0.5f*height+0.5f)*pointDistance;
	float d = sqrtf((pos.x-editX)*(pos.x-editX) + (pos.y-editY)*(pos.y-editY));
	float amount=0.0f;
	if(d<maxDist)
	{
	amount = editValue*(1.0-smoothstep(0.0f,maxDist,d)*1.0f)*dt;
	}
	out[y*width+x] = max(in[y*width+x]+ amount,0.0f);
}	
//Requires dTexWaterH dTexWaterRainRate
__global__ void kernelIncWater (float* out, float4* outBuffer, size_t width, unsigned int height,int rain)
{
	
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;		

		
	//out[y*width+x] = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height) + dCoef_Timestep*max(0.0f,dCoef_RainRate*min(tex2D(dTexWaterRainRate,(x+0.5f+sinf(mobileOffset*100.0f)*40.f)/height,(y+0.5f)/height)+mobileOffset*3.2f,1.0f));
	//out[y*width+x] = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height) + dCoef_Timestep*max(0.0f,dCoef_RainRate*tex2D(dTexWaterRainRate, (x+0.5f+50.f*__sinf(0.001f*(float)clock()))/height, (y+0.5f)/height));
	if(rain)
	{
		out[y*width+x] = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height) +
						 dCoef_Timestep*max(0.0f,dCoef_RainRate*tex2D(dTexWaterRainRate, (x+0.5f)/height, (y+0.5f)/height));
	}
	else out[y*width+x] = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height);
	
	//outBuffer[y*width+x].x=tex2D(dTexWaterRainRate,(x+0.5f)/height,(y+0.5f)/height);
	//outBuffer[y*width+x].y=tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height);
	//outBuffer[y*width+x].z=tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height) + dCoef_Timestep*dCoef_RainRate*tex2D(dTexWaterRainRate,(x+0.5f)/height,(y+0.5f)/height);
	//outBuffer[y*width+x].w=0.0f;
}
//Requires dTexWaterH dTexTerrainH dTexWaterFlux
__global__ void kernelCalculateFlux(float4* out, float4* outBuffer, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	
	//x - left, y -right, z - top, w - bottom
	
	float coefficients=dCoef_Timestep*dCoef_PipeCrossSection*dCoef_G/dCoef_PipeLength;
	float localWaterH = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height);
	float totalLocalH = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height)+localWaterH;

	float hDifferenceL = totalLocalH -tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f)/height)-tex2D(dTexWaterH,(x+0.5f-1.0f)/height,(y+0.5f)/height);
	float hDifferenceR = totalLocalH -tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f)/height)-tex2D(dTexWaterH,(x+0.5f+1.0f)/height,(y+0.5f)/height);
	float hDifferenceT = totalLocalH -tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f-1.0f)/height)-tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f-1.0f)/height);
	float hDifferenceB = totalLocalH -tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f+1.0f)/height)-tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f+1.0f)/height);
	
	//if (abs(hDifferenceL)<0.15f) hDifferenceL =0.0f;
	//if (abs(hDifferenceR)<0.15f) hDifferenceR =0.0f;
	//if (abs(hDifferenceT)<0.15f) hDifferenceT =0.0f;
	//if (abs(hDifferenceB)<0.15f) hDifferenceB =0.0f;

	//My Modification.
	//Conserves the previous flux, if negative flux for a direction is necessary, subtracts no more than 0.01 THIS COULD BE A PROBLEM!
	//Maybe I should subtract 0.01 only if the result would be negative?
	//Has problems with paper formula for the factor. Doesnt allow timestep of 0.2. Which is possible with my formula for the factor
	/*float fluxL = max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).x + max(-0.01, coefficients*hDifferenceL));
	float fluxR = max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).y + max(-0.01, coefficients*hDifferenceR));
	float fluxT = max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).z + max(-0.01, coefficients*hDifferenceT));
	float fluxB = max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).w + max(-0.01, coefficients*hDifferenceB));*/

	//My Modification. Version with ifs.
	//Improved to allow diminishing flux, while preventing negative values and too quick decrementation
	/*float fluxL = tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).x + coefficients*hDifferenceL;
	if (fluxL<0.0) fluxL = max(0.0,tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).x-0.01);
	float fluxR = tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).y + coefficients*hDifferenceR;
	if (fluxR<0.0) fluxR = max(0.0,tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).y-0.01);
	float fluxT = tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).z + coefficients*hDifferenceT;
	if (fluxT<0.0) fluxT = max(0.0,tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).z-0.01);
	float fluxB = tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).w + coefficients*hDifferenceB;
	if (fluxB<0.0) fluxB = max(0.0,tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).w-0.01);*/

	//My solution. Doesn't work with the new factor equation. Also results in wrong results!!!
	/*float fluxL = max(tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).x,max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).x + coefficients*hDifferenceL));
	float fluxR = max(tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).y,max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).y + coefficients*hDifferenceR));
	float fluxT = max(tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).z,max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).z + coefficients*hDifferenceT));
	float fluxB = max(tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).w,max(0.0, tex2D(dTexWaterFlux,(x+0.5)/height,(y+0.5)/height).w + coefficients*hDifferenceB));*/

	//Original version. Best. Caution with the timestep. 0.05 works well
	float fluxL = max(0.0f, tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f)/height).x + coefficients*hDifferenceL);
	float fluxR = max(0.0f, tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f)/height).y + coefficients*hDifferenceR);
	float fluxT = max(0.0f, tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f)/height).z + coefficients*hDifferenceT);
	float fluxB = max(0.0f, tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f)/height).w + coefficients*hDifferenceB);

	float totalFlux=(fluxL+fluxR+fluxT+fluxB)*dCoef_Timestep;
	
	float localWaterVolume = localWaterH*dCoef_PipeLength*dCoef_PipeLength;

	float factor=.999f;
	if(totalFlux>localWaterVolume)
	{
		//Mei's formula for the factor 
		factor = min(1.0f, localWaterVolume/(totalFlux));
		//factor = (localWaterH*dCoef_Timestep/totalFlux);		
	}
	
	out[y*width+x].x = fluxL*factor;
	out[y*width+x].y = fluxR*factor;
	out[y*width+x].z = fluxT*factor;
	out[y*width+x].w = fluxB*factor;
	//outBuffer[y*width+x].x = totalLocalH;
	//outBuffer[y*width+x].y = fluxL*factor;
	//outBuffer[y*width+x].z = fluxL*factor+fluxR*factor+fluxT*factor+fluxB*factor;
	//outBuffer[y*width+x].w = localWaterH;

}

//Requires dTexTerrainH dTexHardness dTexThermalAmnt2Move
__global__ void kernelThermalErosionFlux (float4* out, float4* outDiag, size_t width, unsigned int height, float4* outDebugBuffer)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
	float4 talus;
	float4 talusD;
	float4 hDiff;
	float4 hDiffD;
	float4 outTemp;
	float4 outDiagTemp;
	float localH = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height);
	float total=0.0f;
	hDiff.x = localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f)/height);
	total+=max(hDiff.x,0.0f);
	hDiff.y = localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f)/height);
	total+=max(hDiff.y,0.0f);
	hDiff.z = localH-tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f-1.0f)/height);
	total+=max(hDiff.z,0.0f);
	hDiff.w = localH-tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f+1.0f)/height);
	total+=max(hDiff.w,0.0f);
	hDiffD.x = localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f-1.0f)/height);
	total+=max(hDiffD.x,0.0f);
	hDiffD.y = localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f+1.0f)/height);
	total+=max(hDiffD.y,0.0f);
	hDiffD.z = localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f-1.0f)/height);
	total+=max(hDiffD.z,0.0f);
	hDiffD.w = localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f+1.0f)/height); 
	total+=max(hDiffD.w,0.0f);
	
	//Take hardness into account
	total = total/(1.0f-max(dCoef_HardnessMin,tex2D(dTexHardness,(x+0.5f)/height,(y+0.5f)/height)));
	talus=hDiff/dCoef_PipeLength;
	
	float diagDistance = sqrtf(dCoef_PipeLength*dCoef_PipeLength+dCoef_PipeLength*dCoef_PipeLength);
	talusD=hDiffD/diagDistance;	
	float coef = dCoef_talusRatio+dCoef_talusBias;
	
	//Cheap chemical erosion
	float2 velocity = tex2D(dTexWaterVelocity,(x+0.5f)/height,(y+0.5f)/height);
	float waterH = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height);
	if(waterH>dCoef_DepthMax)
	coef = max(0.1f,coef-max(0.5f*dCoef_talusRatio,length(velocity)/3.5f));

	float amount = tex2D(dTexThermalAmnt2Move,(x+0.5f)/height,(y+0.5f)/height);
	
		

	if(hDiff.x>0.0f && talus.x>coef)
	outTemp.x=amount*(hDiff.x)/total;
	else
	outTemp.x=0.0f;
	if(hDiff.y>0.0f && talus.y>coef)
	outTemp.y=amount*(hDiff.y)/total;
	else
	outTemp.y=0.0f;
	if(hDiff.z>0.0f && talus.z>coef)
	outTemp.z=amount*(hDiff.z)/total;
	else
	outTemp.z=0.0f;
	if(hDiff.w>0.0f && talus.w>coef)
	outTemp.w=amount*(hDiff.w)/total;
	else
	outTemp.w=0.0f;

	if(hDiff.x>0.0f && talusD.x>coef)
	outDiagTemp.x=amount*(hDiffD.x)/total;
	else
	outDiagTemp.x=0.0f;
	if(hDiff.y>0.0f && talusD.y>coef)
	outDiagTemp.y=amount*(hDiffD.y)/total;
	else
	outDiagTemp.y=0.0f;
	if(hDiff.z>0.0f && talusD.z>coef)
	outDiagTemp.z=amount*(hDiffD.z)/total;
	else
	outDiagTemp.z=0.0f;
	if(hDiff.w>0.0f && talusD.w>coef)
	outDiagTemp.w=amount*(hDiffD.w)/total;
	else
	outDiagTemp.w=0.0f;
	
	
	
	out[y*width+x] = outTemp;
	outDiag[y*width+x] = outDiagTemp;

	//DEBUG
	//outDebugBuffer[y*width+x].z=max(max(out[y*width+x].x,out[y*width+x].y),max(out[y*width+x].z,out[y*width+x].w));
	//outDebugBuffer[y*width+x].w=max(max(outDiag[y*width+x].x,outDiag[y*width+x].y),max(outDiag[y*width+x].z,outDiag[y*width+x].w));

	//debugBuffer[y*width+x].z = amount;
}


//Requires dTexTerrainH dTexThermalFlux dTexThermalFluxDiag
__global__ void kernelThermalDrop (float* out, size_t width, unsigned int height,float4* outDebugBuffer)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	
	float sum=tex2D(dTexThermalFlux,(x+0.5f-1.0f)/height,(y+0.5f)/height).y;
	sum+=tex2D(dTexThermalFlux,(x+0.5f+1.0f)/height,(y+0.5f)/height).x;
	sum+=tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f-1.0f)/height).w;
	sum+=tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f+1.0f)/height).z;
	sum-=tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f)/height).x;
	sum-=tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f)/height).y;
	sum-=tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f)/height).z;
	sum-=tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f)/height).w;
	
	sum+=tex2D(dTexThermalFluxDiag,(x+0.5f-1.0f)/height,(y+0.5f-1.0f)/height).y;
	sum+=tex2D(dTexThermalFluxDiag,(x+0.5f+1.0f)/height,(y+0.5f+1.0f)/height).x;
	sum+=tex2D(dTexThermalFluxDiag,(x+0.5f-1.0f)/height,(y+0.5f+1.0f)/height).z;
	sum+=tex2D(dTexThermalFluxDiag,(x+0.5f+1.0f)/height,(y+0.5f-1.0f)/height).w;	
	sum-=tex2D(dTexThermalFluxDiag,(x+0.5f)/height,(y+0.5f)/height).x;
	sum-=tex2D(dTexThermalFluxDiag,(x+0.5f)/height,(y+0.5f)/height).y;
	sum-=tex2D(dTexThermalFluxDiag,(x+0.5f)/height,(y+0.5f)/height).z;
	sum-=tex2D(dTexThermalFluxDiag,(x+0.5f)/height,(y+0.5f)/height).w;
	float result = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height) + sum;
	out[y*width+x] = result; //result;
	//outDebugBuffer[y*width+x].z = (tex2D(dTexThermalFlux,(x+0.5f-1.0f)/height,(y+0.5f)/height).y+tex2D(dTexThermalFlux,(x+0.5f+1.0f)/height,(y+0.5f)/height).x-tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f)/height).x-tex2D(dTexThermalFlux,(x+0.5f)/height,(y+0.5f)/height).y)*0.5f;
	//outDebugBuffer[y*width+x].w = 1.0f;
	//outBuffer[y*width+x].z = result;
	//outBuffer[y*width+x].z = tex2D(dTexTerrainH,x+0.5f,y+0.5f);
	
}
//Requires dTexWaterFlux dTexWaterH dTexTerrainH
__global__ void kernelFlow (float2* outVelocity,float* outSedCap,float* outWH, float4* outFlux,float4* outBuffer, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	
	//Velocity Field
	float2 velocity;
	float4 localFlux = tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f)/height);
	float4 neighbourFlux; 
	neighbourFlux.x = tex2D(dTexWaterFlux,(x+0.5f-1.0f)/height,(y+0.5f)/height).y;
	neighbourFlux.y = tex2D(dTexWaterFlux,(x+0.5f+1.0f)/height,(y+0.5f)/height).x;
	neighbourFlux.z = tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f-1.0f)/height).w;
	neighbourFlux.w = tex2D(dTexWaterFlux,(x+0.5f)/height,(y+0.5f+1.0f)/height).z;

	float waterDelta = neighbourFlux.x + neighbourFlux.y + neighbourFlux.z + neighbourFlux.w -				   
					   localFlux.x - localFlux.y - localFlux.z - localFlux.w;

	//Velocity calculations
	velocity.x=(neighbourFlux.x-neighbourFlux.y-localFlux.x+localFlux.y)*0.5f;
	velocity.y=(neighbourFlux.z-neighbourFlux.w-localFlux.z+localFlux.w)*0.5f;		
	outVelocity[y*width+x] = velocity;	
	float velocityLength=length(velocity);
	
	//Calculate flow (final water height)
	float waterH = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height);
	waterH = waterH + (waterDelta*dCoef_Timestep)/(dCoef_PipeLength*dCoef_PipeLength);
	outWH[y*width+x] = waterH;
	
	//Sedimemt Capacity
	

	
	//Limiter, actually increases erosion as depth decreases
	float limit = max(0.0f,1.0f-waterH/dCoef_DepthMax);
	
	//Solution A
	//outSedCap[y*width+x] = max(0.3f,sinf(atanf( max(0.2f,0.5f+tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height) - tex2D(dTexTerrainH,(x+0.5f+normalize(velocity).x)/height,(y+0.5f+normalize(velocity).y/height)))/(1.5f*dCoef_PipeLength))))*min(3.f0,velocityLength*(limit+1.0f))*0.2f;
	
	
	
	//Solution B

	//Restrict velocity to normalized or 0.0
	velocity = normalize(velocity);
	if (velocityLength<0.2f) velocity.x = velocity.y = 0.0f;
	int count=0;
	float localH = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height);
	float deltaH = localH - tex2D(dTexTerrainH,(x+0.5f+velocity.x)/height,(y+0.5f+velocity.y)/height);
	if (deltaH>-.5f && deltaH < 0.3f)deltaH+=min(velocityLength*0.5f,0.5f*dCoef_PipeLength);	
	/*if(localH<tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f)/height))count++;
	if(localH<tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f)/height))count++;
	if(localH<tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f+1.0f)/height))count++;
	if(localH<tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f-1.0f)/height))count++;
	float factor = 1.0f;
	if (count>2) factor = .1f;*/
	
	/*outSedCap[y*width+x] = min(waterH,
		                      (0.1 + __sinf(atanf(deltaH/dCoef_PipeLength))) * 
							   min(3.0f,velocityLength*(limit+1.0f))*0.5f);*/

	outSedCap[y*width+x] = max(0.0f,0.1f+__sinf(atanf(deltaH/dCoef_PipeLength))*min(3.0f,velocityLength*(limit+1.0f))*0.5f);
	
	
	//outSedCap[y*width+x] = sedimentCapacity;	
	
	
	//outBuffer[y*width+x].z = min(3.0f,velocityLength*(limit+1.0f));
	outBuffer[y*width+x].w = velocityLength;
}
//Requires dTexSedimentAmount dTexSedimentCapacity dTexHardness dTexTerrainH dTexWaterH
__global__ void kernelErodeDepose(float* outHardness, float* outWH, float* outSedAmount, float* outTH, float4* outBuffer, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	float sedAmnt = tex2D(dTexSedimentAmount,(x+0.5f)/height,(y+0.5f)/height);
	float sedCap  = tex2D(dTexSedimentCapacity,(x+0.5f)/height,(y+0.5f)/height);
	float hardness =tex2D(dTexHardness,(x+0.5f)/height,(y+0.5f)/height);
	float factor = dCoef_Timestep*(sedCap-sedAmnt);
	if(sedCap>sedAmnt)
	{
		//factor = min(0.1*dCoef_Timestep,factor);
		factor *= dCoef_DissolveRate*(1.0f-max(dCoef_HardnessMin,hardness));		 		 
		//factor = (sedCap-sedAmnt)*0.1f;
		outHardness[y*width+x] = max(dCoef_HardnessMin, hardness - dCoef_Timestep*dCoef_SoftenRate);
	}
	else 
	{
		//factor = max(-0.1f*dCoef_Timestep,factor);
		factor *= dCoef_SedimentDropRate; 		
		//factor = -0.2f*sedAmnt;
		//factor=1.0f;	
		//outHardness[y*width+x] = min(0.8f, hardness + dCoef_Timestep*dCoef_SoftenRate*0.2f);
	}
	
	if (hardness<-2.f) 
	{
			outHardness[y*width+x] = .80f;
	}
	
	
	//Mass preservation!
	if (factor>0)factor = min(factor,tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height));

	outTH[y*width+x] = max(0.0f,tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height) - factor);	
	outSedAmount[y*width+x] = max(0.0f, tex2D(dTexSedimentAmount,(x+0.5f)/height,(y+0.5f)/height) + factor);
	outWH[y*width+x] = tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height);
	//outWH[y*width+x] = max(0.0f, tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height) + factor);

	
	/*if(sedCap>sedAmnt)
	{
		if (hardness==dCoef_HardnessMin) 
		{
			outHardness[y*width+x] = .80f;
		}
		else
		{
			outHardness[y*width+x] = max(dCoef_HardnessMin, hardness - dCoef_Timestep*dCoef_SoftenRate);
		}
	}*/
	//outHardness[y*width+x] = 0.0f;

	outBuffer[y*width+x].x = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height) - factor;
	outBuffer[y*width+x].y = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height) - factor + max(0.0f, tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height) + factor);
	//outBuffer[y*width+x].z = hardness;	
	outBuffer[y*width+x].z = sedAmnt;
	//outBuffer[y*width+x].z = sedCap;
	//outBuffer[y*width+x].w = dCoef_Timestep*(sedCap-sedAmnt);
}

//__global__ void kernelFixAdvectSediment(float* out, float amountToAdd, size_t width, unsigned int height)
//{
//	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//	
//	out[y*width+x] = tex2D(dTexSedimentAmount,(x+0.5f)/height,(y+0.5f)/height) + amountToAdd*0.0f;
//}

//Requires dTexWaterVelocity dTexSedimentAmount
__global__ void kernelAdvectSediment(float* out, float4* outDebug, size_t width, unsigned int height)
{	
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	float2 velocity = tex2D(dTexWaterVelocity,(x+0.5f)/height,(y+0.5f)/height);
	out[y*width+x] = tex2D(dTexSedimentAmount,(x+0.5f-velocity.x*dCoef_Timestep/dCoef_PipeLength)/height,(y+0.5f-velocity.y*dCoef_Timestep/dCoef_PipeLength)/height);
}
//Requires dTexWaterVelocity dTexSedimentAmntAdvect
__global__ void kernelAdvectBackSediment(float* out, float4* outDebug, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	float2 velocity = tex2D(dTexWaterVelocity,(x+0.5f)/height,(y+0.5f)/height);
	float error = tex2D(dTexSedimentAmount,(x+0.5f)/height,(y+0.5f)/height) - 
		          tex2D(dTexSedimentAmntAdvect,(x+0.5f+velocity.x*dCoef_Timestep/dCoef_PipeLength)/height,(y+0.5f+velocity.y*dCoef_Timestep/dCoef_PipeLength)/height);	

	float4 clamps;
	clamps.x = tex2D(dTexSedimentAmount,(x+0.5f-ceilf(velocity.x*dCoef_Timestep/dCoef_PipeLength))/height,(y+0.5f-ceilf(velocity.y*dCoef_Timestep/dCoef_PipeLength))/height);
	clamps.y = tex2D(dTexSedimentAmount,(x+0.5f-floorf(velocity.x*dCoef_Timestep/dCoef_PipeLength))/height,(y+0.5f-floorf(velocity.y*dCoef_Timestep/dCoef_PipeLength))/height);
	clamps.z = tex2D(dTexSedimentAmount,(x+0.5f-floorf(velocity.x*dCoef_Timestep/dCoef_PipeLength))/height,(y+0.5f-ceilf(velocity.y*dCoef_Timestep/dCoef_PipeLength))/height);
	clamps.w = tex2D(dTexSedimentAmount,(x+0.5f-ceilf(velocity.x*dCoef_Timestep/dCoef_PipeLength))/height,(y+0.5f-floorf(velocity.y*dCoef_Timestep/dCoef_PipeLength))/height);
	float lowClamp = min( min(clamps.x,clamps.y),min(clamps.w,clamps.z));
	float hiClamp = max( max(clamps.x,clamps.y),max(clamps.w,clamps.z));	
	out[y*width+x] = max(lowClamp,min(hiClamp,tex2D(dTexSedimentAmntAdvect, (x+0.5f)/height,(y+0.5f)/height) + error*0.5f));		             
}

// DO NOT USE DEPRECATED!!!!! 
//Might be useful for the BFECC model if i decide to try it
//Requires dTexWaterVelocity dTexSedimentAmount dTexSedimentAmntAdvect dTexSedimentAmntAdvectBack
__global__ void kernelMoveSediment(float* out, float4* outDebug, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	float2 velocity = tex2D(dTexWaterVelocity,(x+0.5f)/height,(y+0.5f)/height);
		
	
	out[y*width+x] = (tex2D(dTexSedimentAmount,(x+0.5f)/height,(y+0.5f)/height)-tex2D(dTexSedimentAmount,(x+0.5f)/height,(y+0.5f)/height));
	
}
//Requires dTexTerrainH dTexHardness
__global__ void kernelThermalErosionAmnt (float* out,float4* outBuffer, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
	float localH = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height);
	float amountToMove = dCoef_PipeLength*dCoef_PipeLength*dCoef_Timestep*dCoef_ThermalErosionRate*(1.0f-tex2D(dTexHardness,(x+0.5f)/height,(y+0.5f)/height))/2.0f;
	
	//float4 hDiff;
	//float4 hDiffD;
	float hDiff=0.0f;
	/*hDiff.x = localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f)/height);
	hDiff.y = localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f)/height);
	hDiff.z = localH-tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f-1.0f)/height);
	hDiff.w = localH-tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f+1.0f)/height);	
	hDiffD.x = localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f-1.0f)/height);
	hDiffD.y = localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f+1.0f)/height);
	hDiffD.z = localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f-1.0f)/height);
	hDiffD.w = localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f+1.0f)/height); */
	
	hDiff=max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f-1.0f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f+1.0f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f-1.0f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f+1.0f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f+1.0f)/height,(y+0.5f-1.0f)/height)) +
		  max(0.0f,localH-tex2D(dTexTerrainH,(x+0.5f-1.0f)/height,(y+0.5f+1.0f)/height));
	/*float maxHeightDifference = max(
								max(max(hDiff.x,hDiff.y),max(hDiff.w,hDiff.w)),
								max(max(hDiffD.x,hDiffD.y),max(hDiffD.z,hDiffD.w)));	*/

	/*out[y*width+x]=amountToMove*abs(maxHeightDifference);*/
	out[y*width+x]=amountToMove*hDiff;
	//outBuffer[y*width+x].x = localH;
	//outBuffer[y*width+x].y = 1.0f-tex2D(dTexHardness,(x+0.5f)/height,(y+0.5f)/height);
	//outBuffer[y*width+x].z = abs(maxHeightDifference);
	//outBuffer[y*width+x].w = hDiffD.w;
}
//Requires dTexWaterH
__global__ void kernelEvaporate(float* outWater, float4* outBuffer, size_t width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;	
	float result = max(tex2D(dTexWaterH,(x+0.5f)/height,(y+0.5f)/height)*(1.0-dCoef_EvaporationRate*dCoef_Timestep),0.0f);
	outBuffer[y*width+x].x = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height);
	outBuffer[y*width+x].y = tex2D(dTexTerrainH,(x+0.5f)/height,(y+0.5f)/height) + result;
	outWater[y*width+x] = result;
}
//Swap pointers
inline void cuda_exchPtrs(void ** ptrA, void ** ptrB)
{
	void * ptrTmp = *ptrA;
	*ptrA = *ptrB;
	*ptrB = ptrTmp;
}

extern "C" void cuda_EditTerrain(float editX, float editY, float pointDistance, float editValue, unsigned int width, unsigned int height ,float maxDist, float dt, int mode )
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
	if (mode)
	{
	printf("edit A");
	kernelEditBuffer<<< grid, block>>>(dWaterH0_ptr, dWaterH1_ptr, editX, editY, pointDistance, editValue, width, height , maxDist, dt );
	cuda_exchPtrs((void **)&dWaterH0_ptr,(void **)&dWaterH1_ptr);
	}
	else
	{
	printf("edit B");
	kernelEditBuffer<<< grid, block>>>(dTerrainH0_ptr, dTerrainH1_ptr, editX, editY, pointDistance, editValue, width, height , maxDist, dt );
	cuda_exchPtrs((void **)&dTerrainH1_ptr,(void **)&dTerrainH0_ptr);
	/*kernelEditBuffer<<< grid, block>>>(dSedimentAmount0_ptr, dSedimentAmount1_ptr, editX, editY, pointDistance, editValue, width, height , maxDist, dt );
	cuda_exchPtrs((void **)&dSedimentAmount0_ptr,(void **)&dSedimentAmount1_ptr);*/
	}
	
}
int rainCount=0;
int rain=0;
float sedimentArray[512*512];
extern "C" void cuda_Simulate(float4* heights, unsigned int width, unsigned int height, float dt, float *in,float*in2)
{
	rainCount++;
	if(rainCount<330)
	rain=1;
	else rain=0;
	if(rainCount>380)
	{rain=1;rainCount=0;}

    dim3 block(32, 2, 1);
    dim3 grid(width / block.x, height / block.y, 1);
	cudaError ERR;
	//Inc Water
	cudaBindTexture2D((size_t *)&dTexWaterHOffset,dTexWaterHptr,(void *)dWaterH0_ptr,&texFloatChannelDesc,512,512,hWaterH0_pitch);
	cudaBindTexture2D((size_t *)&dTexWaterRainRateOffset, dTexWaterRainRatePtr,(void *)dWaterRainRate0_ptr,&texFloatChannelDesc,512,512,hWaterRainRate0_pitch);
	kernelIncWater<<<grid, block>>>(dWaterH1_ptr,heights,width,height,rain);	
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexWaterHptr);
	cudaUnbindTexture(dTexWaterRainRatePtr);
	cuda_exchPtrs((void **)&dWaterH0_ptr,(void **)&dWaterH1_ptr);
	ERR = cudaGetLastError();

	//Calc Flux
	cudaBindTexture2D((size_t *)&dTexTerrainHOffset,dTexTerrainHptr,(void *)dTerrainH0_ptr,&texFloatChannelDesc,512,512,hTerrainH0_pitch);
	cudaBindTexture2D((size_t *)&dTexWaterHOffset,dTexWaterHptr,(void *)dWaterH0_ptr,&texFloatChannelDesc,512,512,hWaterH0_pitch);
	cudaBindTexture2D((size_t *)&dTexWaterFluxOffset,dTexWaterFluxPtr,(void *)dWaterFlux0_ptr,&texFloat4ChannelDesc,512,512,hWaterFlux_pitch);

	kernelCalculateFlux<<<grid,block>>>(dWaterFlux1_ptr,
										heights,
										width, 
										height);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexWaterFluxPtr);
	cuda_exchPtrs((void **)&dWaterFlux0_ptr,(void **)&dWaterFlux1_ptr);
	ERR = cudaGetLastError();
	//cudaUnbindTexture(dTexTerrainHptr);//barrier
	//cudaUnbindTexture(dTexWaterHptr);//barrier

	//Calc Errode Amnt
	cudaBindTexture2D(  (size_t *)&dTexHardnessOffset,
						dTexHardnessPtr,
						(void *)dHardness0_ptr,
						&texFloatChannelDesc,
						512,512,
						hHardness0_pitch);

	kernelThermalErosionAmnt<<<grid,block>>>(	dThermalAmount2Move_ptr,
												heights,
												width,
												height);
	cudaDeviceSynchronize();
	ERR = cudaGetLastError();
	//cudaUnbindTexture(dTexHardnessPtr);//barrier
	//Calculate Flow
	cudaBindTexture2D(	(size_t *)&dTexWaterFluxOffset,
						dTexWaterFluxPtr,
						(void *)dWaterFlux0_ptr,
						&texFloat4ChannelDesc,
						512,512,
						hWaterFlux_pitch);

	kernelFlow<<<grid,block>>>( dWaterVelocity_ptr,
								dSedimentCapacity_ptr,
								dWaterH1_ptr,
								dWaterFlux1_ptr,
								heights, 
								width, 
								height);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexWaterHptr);
	cudaUnbindTexture(dTexWaterFluxPtr);
	cuda_exchPtrs((void **)&dWaterH0_ptr,(void **)&dWaterH1_ptr);
	//cuda_exchPtrs((void **)&dWaterFlux0_ptr,(void **)&dWaterFlux1_ptr);
	ERR = cudaGetLastError();
	//cudaUnbindTexture(dTexTerrainHptr);//barrier
	//cudaUnbindTexture(dTexHardnessPtr);//barrier

	//Calculate ErodeDepose
	cudaBindTexture2D(
						(size_t *)&dTexWaterHOffset,
						dTexWaterHptr,
						(void *)dWaterH0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hWaterH0_pitch);
	cudaBindTexture2D(
						(size_t *)&dTexSedimentAmountOffset,
						dTexSedimentAmountPtr,
						(void *)dSedimentAmount0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hSedimentAmount_pitch);
	cudaBindTexture2D(
						(size_t *)&dTexSedimentCapacityOffset,
						dTexSedimentCapacityPtr,
						(void *)dSedimentCapacity_ptr,
						&texFloatChannelDesc,
						512,512,
						hSedimentCapacity_pitch);

	kernelErodeDepose<<<grid,block>>>(dHardness1_ptr,dWaterH1_ptr,dSedimentAmount1_ptr,dTerrainH1_ptr,heights,width, height);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexWaterHptr);
	cudaUnbindTexture(dTexHardnessPtr);
	cudaUnbindTexture(dTexTerrainHptr);
	cudaUnbindTexture(dTexSedimentAmountPtr);	
	cudaUnbindTexture(dTexSedimentCapacityPtr);	
	cuda_exchPtrs((void **)&dWaterH0_ptr,(void **)&dWaterH1_ptr);
	cuda_exchPtrs((void **)&dHardness0_ptr,(void **)&dHardness1_ptr);
	cuda_exchPtrs((void **)&dTerrainH0_ptr,(void **)&dTerrainH1_ptr);
	
	cuda_exchPtrs((void **)&dSedimentAmount0_ptr,(void **)&dSedimentAmount1_ptr);	
//	ERR = cudaGetLastError();
	//Move Sedmient
	cudaBindTexture2D(
						(size_t *)&dTexWaterHOffset,
						dTexWaterHptr,
						(void *)dWaterH0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hWaterH0_pitch);
	cudaBindTexture2D(
						(size_t *)&dTexSedimentAmountOffset,
						dTexSedimentAmountPtr,
						(void *)dSedimentAmount0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hSedimentAmount_pitch);

	cudaBindTexture2D(
						(size_t *)&dTexWaterVelocityOffset,
						dTexWaterVelocityPtr,
						(void *)dWaterVelocity_ptr,
						&texFloat2ChannelDesc,
						512, 512,
						hWaterVelocity_pitch);
	kernelAdvectSediment<<<grid,block>>>(dSedimentAmntAdvect_ptr, heights, width, height);
	cudaDeviceSynchronize();

	cudaBindTexture2D(
                        (size_t *)&dTexSedimentAmntAdvectOffset,
						dTexSedimentAmntAdvectPtr,
						(void *)dSedimentAmntAdvect_ptr,
						&texFloatChannelDesc,
						512, 512,
						hSedimentAmount_pitch);
	kernelAdvectBackSediment<<<grid,block>>>(dSedimentAmount1_ptr, heights, width, height);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexSedimentAmountPtr);
	cudaUnbindTexture(dTexSedimentAmntAdvectPtr);	
	cuda_exchPtrs((void **)&dSedimentAmount0_ptr,(void **)&dSedimentAmount1_ptr);
	
	ERR = cudaGetLastError();
	
	//Errode thermally
	cudaBindTexture2D(
						(size_t *)&dTexHardnessOffset,
						dTexHardnessPtr,
						(void *)dHardness0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hHardness0_pitch);
	cudaBindTexture2D(
						(size_t *)&dTexThermalAmnt2MoveOffset,
						dTexThermalAmnt2MovePtr,
						(void *)dThermalAmount2Move_ptr,
						&texFloatChannelDesc,
						512, 512,
						hThermalAmount2Move_pitch);
	cudaBindTexture2D(
						(size_t *)&dTexTerrainHOffset,
						dTexTerrainHptr,
						(void *)dTerrainH0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hTerrainH0_pitch);

	kernelThermalErosionFlux<<<grid,block>>>(dThermalFlux_ptr,dThermalFluxDiag_ptr,width,height, heights);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexThermalAmnt2MovePtr);
	cudaUnbindTexture(dTexHardnessPtr);
	cudaUnbindTexture(dTexWaterVelocityPtr);
	ERR = cudaGetLastError();

//	//Drop Thermal Eroded
	cudaBindTexture2D(
						(size_t *)&dTexThermalFluxOffset,
						dTexThermalFluxPtr,
						(void *)dThermalFlux_ptr,
						&texFloat4ChannelDesc,
						512,512,
						hThermalFlux_pitch);
	cudaBindTexture2D(
						(size_t *)&dTexThermalFluxDiagOffset,
						dTexThermalFluxDiagPtr,
						(void *)dThermalFluxDiag_ptr,
						&texFloat4ChannelDesc,
						512,
						512,
						hThermalFluxDiag_pitch);

	kernelThermalDrop<<<grid,block>>>(dTerrainH1_ptr,width,height,heights);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexTerrainHptr);
	cudaUnbindTexture(dTexThermalFluxPtr);
	cudaUnbindTexture(dTexThermalFluxDiagPtr);	
	cuda_exchPtrs((void **)&dTerrainH0_ptr,(void **)&dTerrainH1_ptr);
//ERR = cudaGetLastError();

	//Evaporate
	cudaBindTexture2D(
						(size_t *)&dTexTerrainHOffset,
						dTexTerrainHptr,
						(void *)dTerrainH0_ptr,
						&texFloatChannelDesc,
						512, 512,
						hTerrainH0_pitch);
	//cudaBindTexture2D((size_t *)&dTexWaterHOffset,dTexWaterHptr,(void *)dWaterH0_ptr,&texFloatChannelDesc,512,512,hWaterH0_pitch);
	kernelEvaporate<<<grid,block>>>(dWaterH1_ptr, heights,width, height);
	cudaDeviceSynchronize();
	cudaUnbindTexture(dTexWaterHptr);
	cudaUnbindTexture(dTexTerrainHptr);
	cuda_exchPtrs((void **)&dWaterH0_ptr,(void **)&dWaterH1_ptr);
	ERR = cudaGetLastError();
	
//cudaBindTexture2D((size_t *)&dTexHardnessOffset,dTexTerrainHptr,(void *)dTerrainH0_ptr,&texFloatChannelDesc,512,512,hTerrainH0_pitch);
//	cudaBindTexture2D((size_t *)&dTexWaterRainRateOffset, dTexWaterRainRatePtr,(void *)dWaterRainRate0_ptr,&texFloatChannelDesc,512,512,hWaterRainRate0_pitch);//bound=1;
//    kernelTest<<< grid, block>>>(heights, width, height, dt, dTerrainH1_ptr, dWaterRainRate1_ptr);
//	cudaUnbindTexture(dTexTerrainHptr);
//	cudaUnbindTexture(dTexWaterRainRatePtr);
//	cuda_exchPtrs((void **)&dTerrainH0_ptr,(void **)&dTerrainH1_ptr);
//	cuda_exchPtrs((void **)&dWaterRainRate0_ptr,(void **)&dWaterRainRate1_ptr);

	//tick^=1;
}

//extern "C" void cuda_Simulate(float4* heights, unsigned int width, unsigned int height, float dt, float *in,float*in2)
//{
//
//    dim3 block(8, 8, 1);
//    dim3 grid(width / block.x, height / block.y, 1);
//	if(!tick)
//	{
//	cudaBindTexture2D((size_t *)&dTexHardnessOffset,dTexTerrainHptr,(void *)dTerrainH0_ptr,&texFloatChannelDesc,512,512,hTerrainH0_pitch);
//	cudaBindTexture2D((size_t *)&dTexWaterRainRateOffset, dTexWaterRainRatePtr,(void *)dWaterRainRate0_ptr,&texFloatChannelDesc,512,512,hWaterRainRate0_pitch);//bound=1;
//    kernelTest<<< grid, block>>>(heights, width, height, dt, dTerrainH1_ptr, dWaterRainRate1_ptr);
//	cudaUnbindTexture(dTexTerrainHptr);
//	cudaUnbindTexture(dTexWaterRainRatePtr);
//	}
//	else
//	{
//	cudaBindTexture2D((size_t *)&dTexHardnessOffset,dTexTerrainHptr,(void *)dTerrainH1_ptr,&texFloatChannelDesc,512,512,hTerrainH1_pitch);
//	cudaBindTexture2D((size_t *)&dTexWaterRainRateOffset, dTexWaterRainRatePtr,(void *)dWaterRainRate1_ptr,&texFloatChannelDesc,512,512,hWaterRainRate0_pitch);
//    kernelTest<<< grid, block>>>(heights, width, height, dt, dTerrainH0_ptr,dWaterRainRate0_ptr);
//	cudaUnbindTexture(dTexTerrainHptr);
//	cudaUnbindTexture(dTexWaterRainRatePtr);
//	}
//	tick^=1;
//}


//extern "C" void cuda_Simulate(int* heights, unsigned int width, unsigned int height, float dt)
//{
//	//printf("\nCuda Start");
//
//    // execute the kernel
//    dim3 block(8, 8, 1);
//    dim3 grid(width / block.x, height / block.y, 1);
//    kernelTest<<< grid, block>>>(heights, width, height, dt);
//
//	
//	//printf("\nCuda End");
//}
