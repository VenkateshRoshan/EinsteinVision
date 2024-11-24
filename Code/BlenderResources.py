
import bpy
import os
from utils.roadlanelib import *

class BlenderSceneGenerator:
    def __init__(self):
        self.delete_all_objects()
        # self.camera = None  # Initialize camera as None
        self.setup_camera()  # Setup camera
        self.objects = self.load_blend_files()

    def setup_camera(self):
        # Check if camera exists, otherwise create one
        if not bpy.data.objects.get('Camera'):
            bpy.ops.object.camera_add()
        if not bpy.data.objects.get('SUN'):
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
        self.camera = bpy.data.objects.get('Camera')
        self.camera.location = (0, -5, 1.5)
        self.camera.rotation_euler = (1.57, 0, 0)

    def delete_all_objects(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def delete_all_objects_except_camera(self):
        bpy.ops.object.select_all(action='DESELECT')
        if self.camera:
            self.camera.select_set(True)
        bpy.ops.object.delete()
        
    def delete_from_scene(self,frame_num) :
        scene = bpy.context.scene
        for obj in scene.objects :
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.delete()
#            obj.keyframe_insert(data_path="hide_render", frame=frame_num)
#            obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
#            obj.hide_render = True
#            obj.hide_viewport = True
        
    def load_blend_files(self):
        vPath = '/home/abven/CV_Course/badari_p3/EinstienVision/Assets/Vehicles/'
        ePath = '/home/abven/CV_Course/badari_p3/EinstienVision/Assets/Entities/'
        mePath = '/home/abven/CV_Course/badari_p3/EinstienVision/Assets/Mod_Entities/'
        mVPath = '/home/abven/CV_Course/badari_p3/EinstienVision/Assets/Vehicles_RedLight/'
        
        # v_objects, en_objects = [], []
        objects = []
        exclude_Objects = ['Camera', 'Sun', 'Light' , 'Cube']
        
        for bFile in sorted(os.listdir(vPath)):
            with bpy.data.libraries.load(vPath + '/' + bFile) as (data_from, vehicle_data):
                for obj_from in data_from.objects:
                    if obj_from not in exclude_Objects:
                        # print(obj_from)
                        vehicle_data.objects.append(obj_from)
            for veh in vehicle_data.objects:
                objects.append(veh)

        for bFile in sorted(os.listdir(ePath)) :
            with bpy.data.libraries.load(ePath + '/' + bFile) as (data_from, ent_data):
                for obj_from in data_from.objects:
                    if obj_from not in exclude_Objects:
                        # print(obj_from)
                        ent_data.objects.append(obj_from)
            for en in ent_data.objects:
                objects.append(en)

        for bFile in sorted(os.listdir(mePath)) :
            with bpy.data.libraries.load(mePath + '/' + bFile) as (data_from, ent_data):
                for obj_from in data_from.objects:
                    if obj_from not in exclude_Objects:
                        # print(obj_from)
                        ent_data.objects.append(obj_from)
            for en in ent_data.objects:
                objects.append(en)

        for bFile in sorted(os.listdir(mVPath)) :
            with bpy.data.libraries.load(mVPath + '/' + bFile) as (data_from, ent_data):
                for obj_from in data_from.objects:
                    if obj_from not in exclude_Objects:
                        # print(obj_from)
                        ent_data.objects.append(obj_from)
            for en in ent_data.objects:
                objects.append(en)

        return objects
    
    def load_objects_into_frame(self, objects, orientations=None, locations=None, scales=None, frame_number=1,camera_loc=(0,0,1.5),ObjectState=None):
        
        bpy.context.scene.frame_set(frame_number)
        self.setup_camera()
        self.setup_camera_location(camera_loc)

        for obj_c, obj in enumerate(objects):
            
            obj = obj.copy()
            obj.data = obj.data.copy()
            if ObjectState is not None:
                obj = addTexture(obj)
            obj.rotation_euler = orientations[obj_c]
            obj.location = locations[obj_c]
            obj.scale = scales[obj_c]
            obj.keyframe_insert(data_path="location", frame = frame_number ,index=-1)
            obj.keyframe_insert(data_path="rotation_euler", frame = frame_number , index=-1)
            obj.keyframe_insert(data_path="scale", frame = frame_number , index=-1)
            
            bpy.context.collection.objects.link(obj)
        del objects, orientations, locations, scales

    def setup_camera_location(self,camera_loc):
        self.camera.location = camera_loc

    def render_frame(self, output_path, frame_name, frame_num, width=1920, height=1080):
        # add light to the scene on top of the camera
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
        bpy.context.scene.camera = self.camera
        bpy.context.scene.render.filepath = os.path.join(output_path, f"{frame_name}")
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.ops.render.render(write_still=True)
        self.delete_from_scene(frame_num)

def addTexture(obj,Color=(0,0.1,0,0)):
    matg = bpy.data.materials.new("Green")
    matg.diffuse_color = Color
    obj.active_material = matg

    return obj


def main() :
    generator = BlenderSceneGenerator()
    print(generator.objects)

    obj_name = 'Car'
    # obj = bpy.data.objects[obj_name]
    obj = generator.objects[4]

    #====================== Colors =========================
    # Green
    matg = bpy.data.materials.new("Green")
    matg.diffuse_color = (0,0.1,0,0.8)
    # Red
    matr = bpy.data.materials.new("red")
    matr.diffuse_color = (0.01,0,0,0)

    obj.active_material = matr

    generator.load_objects_into_frame([obj], [(0,0,0)], [(0,0,0)], [(.01,.01,.01)], 1,(0,-5,1))
    generator.render_frame('Outputs/','frame_1.jpg',1)

if __name__ == "__main__":
    main()