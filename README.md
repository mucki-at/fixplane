simple addon to make a polygon in blender planar. There are two changes from loop_tools.make_plane:

1. Better algorithm for finding desired plane:
   1. Get a rough estimate for desired plane (currently picks cartesian plane with biggest projection of polygon).
   2. Perform linear regression
   3. Identify outliers and repeat linear regression with rest.
  
   The idea is that if only a small number of the total vertices in a polygon are not in a plane then we should keep the majority where they are and only fix the outliers.

2. Analyze the neighborhood for each vertex and keep adjacent polygons planar. By limiting the degrees of freedom a vertex can move we can avoid breaking planarity of neighboring faces. Since triangles are always planar we first try to solve the constraints using quads and other polygons and only constrain to triangle planes if we are underdefined.
