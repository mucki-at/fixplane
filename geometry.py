import math
from re import X
import bmesh
import mathutils
import numpy as np
from enum import Enum

class WorkFrameType(Enum):
    """
    Enum to define the type of work plane to be used.
    """
    XY = 1
    XZ = 2
    YZ = 3
    CARTESIAN = 4
    NORMAL = 5

class BestFitType(Enum):
    """
    Enum to define the type of best fit plane to be used.
    """
    NORMAL = 1
    REGRESSION = 2

def determine_work_frame(face: bmesh.types.BMFace, type:WorkFrameType = WorkFrameType.CARTESIAN) -> tuple[mathutils.Vector, mathutils.Matrix]:
    """
    Determine the work plane based on the face's vertices. The
    work plane is roughly aligned with the faces vertices and can be used
    as a safe plane for regression.
    
    Args:
        face (bmesh.types.BMFace): The face to determine the work plane for.
        type (WorkFrameType): The type of work plane to use. Defaults to WorkFrameType.CARTESIAN.
    
    Returns:
        tuple: A tuple containing the center of the work plane, and the coordinate system of the work plane.
    """
    
    match type:
        case WorkFrameType.XY:
            work_co = mathutils.Vector((0.0, 0.0, 0.0))  # Default to origin
            work_frame = mathutils.Matrix(((1.0, 0.0, 0.0),
                                           (0.0, 1.0, 0.0),
                                           (0.0, 0.0, 1.0)))  # Normal pointing in Z direction
        
        case WorkFrameType.XZ:
            work_co = mathutils.Vector((0.0, 0.0, 0.0))  # Default to origin
            work_frame = mathutils.Matrix(((1.0, 0.0, 0.0),
                                           (0.0, 0.0, 1.0),
                                           (0.0, 1.0, 0.0)))  # Normal pointing in Y direction
        
        case WorkFrameType.YZ:
            work_co = mathutils.Vector((0.0, 0.0, 0.0))  # Default to origin
            work_frame = mathutils.Matrix(((0.0, 1.0, 0.0),
                                           (0.0, 0.0, 1.0),
                                           (1.0, 0.0, 0.0)))  # Normal pointing in X direction
        
        case WorkFrameType.CARTESIAN:
            # calculate projected area of the face on all three cardinal planes and chose plane with the largest area
            area_xy = 0.0
            area_xz = 0.0
            area_yz = 0.0
            work_co = face.calc_center_median()

            v_prev = face.verts[-1]
            for v in face.verts:
                # calculate the area of the face projected onto the XY, XZ and YZ planes
                area_xy += (v.co.x+v_prev.co.x) * (v.co.y - v_prev.co.y)/2  # Trapezoidal rule for area calculation
                area_xz += (v.co.x+v_prev.co.x) * (v.co.z - v_prev.co.z)/2
                area_yz += (v.co.y+v_prev.co.y) * (v.co.z - v_prev.co.z)/2
                v_prev = v

            area_xy = abs(area_xy)
            area_xz = abs(area_xz)
            area_yz = abs(area_yz)

            # Determine the largest area and set the work coordinate and normal accordingly
            if area_xy >= area_xz and area_xy >= area_yz:
                work_co.z = 0.0  # Default to origin in Z
                work_frame = mathutils.Matrix(((1.0, 0.0, 0.0),
                                            (0.0, 1.0, 0.0),
                                            (0.0, 0.0, 1.0)))  # Normal pointing in Z direction
            elif area_xz >= area_xy and area_xz >= area_yz:
                work_co.y = 0.0  # Default to origin in Z
                work_frame = mathutils.Matrix(((1.0, 0.0, 0.0),
                                            (0.0, 0.0, 1.0),
                                            (0.0, 1.0, 0.0)))  # Normal pointing in Y direction
            else:
                work_co.x = 0.0  # Default to origin in Z
                work_frame = mathutils.Matrix(((0.0, 1.0, 0.0),
                                            (0.0, 0.0, 1.0),
                                            (1.0, 0.0, 0.0)))  # Normal pointing in X direction

        case WorkFrameType.NORMAL:
            # Calculate the work coordinate as the average of the first three vertices
            work_co = face.calc_center_median()
            
            # Use the face's normal as the work normal
            n = face.normal.normalized()
            work_frame = mathutils.Matrix.OrthoProjection(n,3)
            work_frame[2] = n

        case _:
            raise ValueError(f"Unknown work plane type: {type}")
    
    return work_co, work_frame

def best_fit_plane(face: bmesh.types.BMFace, type:BestFitType = BestFitType.REGRESSION, max_outliers=0.25) -> tuple[mathutils.Vector, mathutils.Vector]:
    """
    Calculate the best fit plane for the face based on the work coordinate
    and work normal. This function uses a simple regression to find the best
    fit plane.
    
    Args:
        face (bmesh.types.BMFace): The face to calculate the best fit plane for.
        type (BestFitType): The type of best fit plane to use. Defaults to BestFitType.REGRESSION.
        max_outliers (float): Maximum percentage of outlier points to remove during regression. Defaults to 25%.
    
    Returns:
        tuple: A tuple containing the fit coordinate and fit normal.
    """
    
    if type == BestFitType.NORMAL:
        # If the type is NORMAL, we can use the face's normal directly
        fit_co = face.calc_center_median()
        fit_no = face.normal.normalized()
        return fit_co, fit_no

    # for three vertices, we can calculate without guessing 
    if len(face.verts)==3:
        v1, v2, v3 = face.verts[:3]
        fit_co = (v1.co + v2.co + v3.co) / 3.0
        fit_no = (v2.co - v1.co).cross(v3.co - v1.co).normalized()
        return fit_co, fit_no
 
    # four vertices is a special case, because each subset of three vertices is planar.
    # we use the distance of the fourth vertex to the plane defined by the first three vertices and make the one with the smallest distance the outlier
    elif len(face.verts)==4 and max_outliers >= 0.25:
        fit_co = mathutils.Vector((0.0, 0.0, 0.0))
        fit_no = mathutils.Vector((0.0, 0.0, 1.0))
        min_distance = float('inf')

        for i in range(4):
            # Calculate the work coordinate and normal for the first three vertices
            v1, v2, v3 = face.verts[(i+1)%4].co, face.verts[(i+2)%4].co, face.verts[(i+3)%4].co

            work_co = (v1+v2+v3) / 3.0
            work_no = (v2-v1).cross(v3-v1).normalized()
            
            dist_v4 = mathutils.geometry.distance_point_to_plane(face.verts[i].co, work_co, work_no)
            if abs(dist_v4) < min_distance:
                min_distance = abs(dist_v4)
                fit_co = work_co
                fit_no = work_no

    else:
        # For more than four vertices, we will use a regression analysis to find the best fit plane.
        # First we determine a very rough estimate of the best fit plane so we can transform the vertices
        # into pairs of coordinates inside the work plane and a distance to the plane.

        work_co, work_frame = determine_work_frame(face, WorkFrameType.CARTESIAN)

        # Transform the vertices into the work coordinate system
        transformed_verts = [work_frame @ (v.co - work_co) for v in face.verts]
        X = np.array([[v.x, v.y, 1] for v in transformed_verts])
        Y = np.array([v.z for v in transformed_verts])
        # Perform a least squares regression to find the best fit plane
        # The plane equation is of the form z = ax + by + c, where a, b, and c are the coefficients
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(X, Y)
        # coeffs[0] is the slope in x direction, coeffs[1] is the slope in y direction, and coeffs[2] is the intercept

        if rank < 3:
            raise ValueError(f"Failed to perform regression on face {face}. The rank of the matrix is {rank}, but it should be 3. The face is probably degenerate or has too few vertices.")
        
        # TODO: check for outliers and remove them if necessary
        
        # get matrix to transform back to the original coordinate system
        work_frame_inv = work_frame.inverted()

        # The normal vector of the plane is (a, b, -1) in the work coordinate system
        fit_co = work_co + work_frame_inv @ mathutils.Vector((0, 0, coeffs[2]))
        fit_no = work_frame_inv @ mathutils.Vector((coeffs[0], coeffs[1], -1)).normalized()
        
    return fit_co, fit_no

def project_point_to_plane(point: mathutils.Vector, plane_co: mathutils.Vector, plane_no: mathutils.Vector) -> mathutils.Vector:
    """
    Project a point onto a plane defined by a coordinate and a normal vector.
    
    Args:
        point (mathutils.Vector): The point to project.
        plane_co (mathutils.Vector): A point on the plane.
        plane_no (mathutils.Vector): The normal vector of the plane.
    
    Returns:
        mathutils.Vector: The projected point on the plane.
    """
    # Calculate the distance from the point to the plane
    distance = mathutils.geometry.distance_point_to_plane(point, plane_co, plane_no)
    # Project the point onto the plane
    return point - distance * plane_no.normalized()

def move_point_to_planes_intersection(point: mathutils.Vector, planes: list[tuple[mathutils.Vector, mathutils.Vector, int]]) -> mathutils.Vector:
    """
    Intersect multiple planes to find a new vertex position.
    
    Args:
        planes (list): A list of tuples containing the plane coordinates and normals.
    
    Returns:
        mathutils.Vector: The intersection point of the planes.
    """
    if len(planes)==0:
        return point
    if len(planes) == 1:
        # If there is only one plane, project the point onto the plane
        return project_point_to_plane(point, planes[0][0], planes[0][1])

    if len(planes) == 2:
        # If there are two planes, intersect them to find the line of intersection
        a_co, a_no = planes[0]
        b_co, b_no = planes[1]
        edge = mathutils.geometry.intersect_plane_plane(a_co, a_no, b_co, b_no)
        if edge is None or edge[0] is None or edge[1] is None:
            raise ValueError(f"Failed to intersect plane {a_co, a_no} with {b_co,b_no}")
        
        # Now find closest point on the line of intersection to the point
        c, dist = mathutils.geometry.closest_point_on_line(point, edge[0], edge[0]+edge[1])
        if c is None:
            raise ValueError(f"Failed to determine closest point of {point} with {edge[0],edge[1]}")
        return c
    
    if len(planes) == 3:
        # If there are two planes, intersect them to find the line of intersection
        a_co, a_no, _ = planes[0]
        b_co, b_no, _ = planes[1]
        c_co, c_no, _ = planes[2]
        edge = mathutils.geometry.intersect_plane_plane(b_co, b_no, c_co, c_no)
        if edge is None or edge[0] is None or edge[1] is None:
            raise ValueError(f"Failed to intersect plane {b_co, b_no} with {c_co,c_no}")
        c = mathutils.geometry.intersect_line_plane(edge[0], edge[0]+edge[1], a_co, a_no)
        if c is None:
            raise ValueError(f"Failed to intersect edge {edge[0], edge[1]} with plane {a_co,a_no}")
        return c
    
    # If there are more than three planes, we will use a more complex method to find the intersection point.
    target = planes[0]

    # Split the planes into triangle and polygon planes based on the number of vertices
    triangle_planes = [x for x in planes[1:] if x[2] <= 3]
    poly_planes = [x for x in planes[1:] if x[2] > 3]

    # triangle count less, because they will always stay planar. See if we have enough polygon planes to calculate a good intersection point
    if len(poly_planes) <= 2:
        # we either have exactly three poly planes, or we can fill the gap with triangle planes 
        use_planes = [target] + poly_planes + triangle_planes[0:2-len(poly_planes)]
        return move_point_to_planes_intersection(point, use_planes)

    candidates = []
    for idx in range(1, len(poly_planes)-1):
        # go through all adjacent planes and calculate the interaction of target and them
        a_co, a_no, _ = poly_planes[idx]
        b_co, b_no, _ = poly_planes[idx+1]
        edge = mathutils.geometry.intersect_plane_plane(a_co, a_no, b_co, b_no)
        if edge is None or edge[0] is None or edge[1] is None:
            continue
        c = mathutils.geometry.intersect_line_plane(edge[0], edge[0]+edge[1], target[0], target[1])
        if c is None:
            continue
        candidates.append(c)

    if len(candidates)==0:
        return None
    
    #the candidates should be averaged to find the best fit point
    print(f"Found {len(candidates)} candidates for intersection of {len(planes)} planes.")
    print(f"{candidates}")
    if len(candidates) == 1:
        return candidates[0]
    
    # pick candidate which is closest to original point
    candidates.sort(key=lambda x: (x - point).length)
    result=candidates[0]
    print(f"Returning intersection point {result} for {len(planes)} planes.")
    return result