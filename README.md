TerrainGenerator
================

My Diploma Thesis: A CUDA accelerated simulator of hydraulic erosion of terrains.
The results are visualized using OpenGL (using freeglut).

I have implemented a mix of the following 2 papers:
  * "Fast Hydraulic and Thermal Erosion on GPU" by Balazs Jako
  * "Interactive terrain modeling using hydraulic erosion" by O Št'ava, B Beneš, M Brisbin, J Křivánek


As a math library, I have used the code samples for the book 
  * "Interactive Computer Graphics, A top-down approach with OpenGL (Sixth Edition)" by Edward Angel
as well as the cutil library.

I have also ported and modified Ken Perlin's reference implementation of perlin noise to C++.

Warning!
===========
This code is very ugly! Since I am using GLUT, all state is held in global variables!
Soon, I hope, I will rewrite the project, using more modern libraries, with a better architecture.
