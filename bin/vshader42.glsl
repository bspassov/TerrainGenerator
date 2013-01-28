#version 150 

in  vec4 vPosition;
out vec4 data;
out vec4 normals;

uniform mat4 model_view;
uniform mat4 projection;
uniform sampler2D heightsV;
uniform int type;

float random(vec3 pos)
{
 return sin(float((int(pos.x*24.41))*(int(pos.z*213.42))^894712904)+pos.y);
 //return max(abs(sin(pos.x*5000.)),abs(sin(pos.z*5000.)));
}

void main() 
{
		vec4 pos =vPosition;
		vec2 uv = (pos.xz+256.0)/(512.0);
		vec4 sampledH = texture2D(heightsV,uv.xy);
		
	
		data = vPosition;
		
		vec4 generator;
		
		if(type==3)
		{
		 if(pos.y!=0.0)
		 {
		 pos.y=0.0;
		 vec3 normal = normalize(vec3(sampledH.z,1.0,sampledH.w))*2.;
		 pos.xyz+=normal;		 
		 }
		 pos.y+=sampledH.x;
		}
		else
		{
		if (type == 0)
		{
			pos.y+=sampledH.x;
			data.w=1.0;
			generator.x = texture2D(heightsV,uv.xy+vec2(2.0/256.,0.0)).x;
			generator.y = texture2D(heightsV,uv.xy+vec2(-2.0/256.,0.0)).x;
			generator.z = texture2D(heightsV,uv.xy+vec2(0.0,2.0/256.)).x;
			generator.w = texture2D(heightsV,uv.xy+vec2(0.0,-2.0/256.)).x;
		}
		else
		{
			pos.y+=sampledH.y+0.001;data.w=0.0;
			generator.x = texture2D(heightsV,uv.xy+vec2(2.0/256.,0.0)).y;
			generator.y = texture2D(heightsV,uv.xy+vec2(-2.0/256.,0.0)).y;
			generator.z = texture2D(heightsV,uv.xy+vec2(0.0,2.0/256.)).y;
			generator.w = texture2D(heightsV,uv.xy+vec2(0.0,-2.0/256.)).y;
		}		
		}
		normals.xyz=normalize(cross(normalize(vec3(4.0,(generator.x -generator.y)*4.,0.0)),normalize(vec3(0.0,(generator.w -generator.z)*4.,-4.0))));
		
		
		//Fooling around with water (Making it sharper)
		//if(type!=0)
		//pos.xyz-=normals.xyz*vec3(1.1+random(pos.xyz)*0.1,0.1,1.1+random(pos.yxz)*0.1)*0.5;
		
    gl_Position = (projection*model_view*(pos*vec4(1.0,1.0,1.0,1.0)))/vPosition.w;
		
		
		if (type != 0) gl_Position-=vec4(0.0,0.0,0.001,0.0);
		
		
		
		
		
    
		data.xz=uv.xy;
} 

