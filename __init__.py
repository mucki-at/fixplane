# -*- coding: utf-8 -*-
import bpy
import bmesh
import mathutils
from . import geometry

class FixPlaneOperator(bpy.types.Operator):
    bl_idname = "fixplane.fixplane"
    bl_label = "Fix Plane"
    bl_description = "Make selected face planar, while also keeping adjacent planes planar."
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    threshold: bpy.props.FloatProperty(name="Threshold",
                                       description="Ignore vertices which are closer to the plane than this threshold.",
                                       default=0.0001, min=0.0, precision=4)
    fix_normal: bpy.props.BoolProperty(name="Fix Normal",
                                       description="Fix the normal of the face to the best fit plane normal.",
                                       default=True)

    def execute(self, context):
        obj = context.active_object
        me = obj.data

        bm = bmesh.from_edit_mesh(me)
        selected_faces = [f for f in bm.faces if f.select and not f.hide]
        if not selected_faces or len(selected_faces) != 1:
            self.report({'WARNING'}, "Must select exacly one face.")
            return {"CANCELLED"}

        face=selected_faces[0]
        self.report({'INFO'}, f"Operating on {face}.")
        
        if len(face.verts) <= 3:
            return {"FINISHED"}

        fit_co, fit_no = geometry.best_fit_plane(face)
        self.report({'INFO'}, f"Fit coordinate: {fit_co}, Fit normal: {fit_no}.")
        skipped = 0
        for v in face.verts:
            dist = mathutils.geometry.distance_point_to_plane(v.co, fit_co, fit_no)
            if abs(dist) > self.threshold:
                # step one, determine planes of all faces adjacent to the vertex
                planes = [(fit_co, fit_no, len(face.verts))]
                for f in v.link_faces:
                    if f != face:
                        f_fit_co, f_fit_no = geometry.best_fit_plane(f)
                        planes.append((f_fit_co, f_fit_no, len(f.verts)))

                # step two: intersect all planes to find the new vertex position
                newpos = geometry.move_point_to_planes_intersection(v.co, planes)
                self.report({'INFO'},f"moving vertex {v.index} from {v.co} to {newpos} (dist={(newpos-v.co).length})")
                v.co=newpos
            else:
                print(f"ignoring vertex {v} because true distance {dist}<={self.threshold}")
                skipped+=1

        self.report({'INFO'}, f"Skipped {skipped} vertices because of threshold.")
        if self.fix_normal:
            # step three: fix the normal of the face
            face.normal = fit_no
            self.report({'INFO'},f"Fixing normal of face {face.index} to {fit_no}.")


        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, "Plane fixed successfully.")
        return {"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    @classmethod
    def poll(self, context):
        """Check if the operator can be called."""

        if context.mode != 'EDIT_MESH':
            self.poll_message_set("This operator can only be used in Edit Mode.")
            return False
        
        if not context.active_object:
            self.poll_message_set("No active object found.")
            return False

        # Check if the active object is a mesh and if it has selected faces.
        if context.active_object.type != 'MESH':
            self.poll_message_set("This operator can only be used with mesh objects.")
            return False
        
        # TODO: Check if the mesh has selected faces.
        return True
        

def menu_func(self, context):
    self.layout.operator(FixPlaneOperator.bl_idname)

def register():
    bpy.utils.register_class(FixPlaneOperator)
    bpy.types.VIEW3D_MT_edit_mesh_faces.append(menu_func)  # Adds the new operator to the mesh faces menu.
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(menu_func)  # Adds the new operator to the mesh faces menu.

def unregister():
    bpy.utils.unregister_class(FixPlaneOperator)
    bpy.types.VIEW3D_MT_edit_mesh_faces.remove(menu_func)  # Adds the new operator to the mesh faces menu.
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(menu_func)  # Adds the new operator to the mesh faces menu.


    # VIEW3D_MT_edit_mesh_faces 