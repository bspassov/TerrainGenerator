#version 150 

in  vec4 data;
out vec4 fColor;
in vec4 normals;
uniform vec2 mouse;
uniform float worldScale;
uniform float editDistance;
uniform sampler2D heights;
uniform float factor;

float random(vec3 pos)
{
 //return sin(float((int(pos.x*24216.)^45424136)*(int(pos.z*24216.)^38412135)));
 return max(abs(sin(pos.x*5000.)),abs(sin(pos.z*5000.)));
}

void main() 
{ 
	  vec4 h = texture2D(heights,data.xz).xyzw;
		
		
		if(data.w==0.0)
    {fColor = vec4(0.0,0.3,0.6,0.5);
		fColor=fColor*((dot(normalize(vec3(1.0,1.0,0.0)),normals.xyz)))*.4+fColor*0.6;
			
		
		if(h.y-h.x>0.01)fColor.w=min(0.7,0.5+pow(min(5.,h.y-h.x)/5.,3.));	
		if (h.y-h.x<.1)fColor.w=0.05;
		
		
		fColor.xyzw=fColor.xyzw*(1.0-h.w/6.0)+vec4(.93,0.96,1.0,0.8)*h.w/6.0;
		fColor.xy=h.zz*0.5;
		
		}
		else
		{
		float isoLines = min(clamp(abs(sin(h.x*3.14))*10.,0.0,1.0)*0.8,clamp(abs(sin(h.x*3.14*0.1))*18.,0.0,1.0));				
		
    //fColor.x *=random(vec3(data.x,h.x,data.z+h.z*10.));
		//fColor.x=h.z;				
		vec3 grass = vec3(0.48,0.81,0.417);
		vec3 rock = vec3(0.492,0.498,0.500);
		vec3 brown = vec3(0.30,0.17,0.08);
		vec3 sand = vec3(0.847,0.674,0.274);
		float upFactor=dot(normalize(vec3(0.0,1.0,0.0)),normals.xyz)*2.0-0.25;
		upFactor=clamp(upFactor*upFactor,0.0,1.0);
		float mudFactor = 0.0;
		if (h.y-h.x>2.0)mudFactor=.5;
		
		fColor.xyz = (upFactor*grass+(1.0-upFactor)*rock)*(1.-h.z)+sand*h.z;
		fColor.xyz = fColor.xyz*(1.0-mudFactor)+mudFactor*brown;
		
		fColor.xyz=((fColor.xyz*((dot(normalize(vec3(1.0,1.0,0.0)),normals.xyz))))*.6+fColor.xyz*0.4);
		
		if(isoLines<0.2)fColor.xyz*=1.5;
		
		fColor.w=1.0;
		}
		//fColor.xyz=-normals.yyy;
		vec2 worldPos = (data.xz*512.0-256.0)*worldScale-mouse.xy*worldScale;
		if (sqrt(dot(worldPos,worldPos))<editDistance)
	  fColor.xyz+=0.2;

} 

