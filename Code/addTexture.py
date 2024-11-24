
import bpy
import numpy as np
import cv2
from BlenderResources import BlenderSceneGenerator
import argparse

def addTexture(image_path, obj_name):

    generator = BlenderSceneGenerator()
    selected_object = generator.objects[19] # sign_25mph_sign_25mph
    
    # add object to scene
    bpy.context.collection.objects.link(selected_object)

    obj = bpy.context.active_object
    obj.select_set(True)

    # get object with name in the scene
    obj = bpy.data.objects['sign_25mph_sign_25mph']

    # # print(dir(obj))

    # mat = bpy.data.materials.new(name="My Material")

    # mat.use_nodes = True
    # bsdf = mat.node_tree.nodes["Principled BSDF"]
    # texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    # texImage.image = bpy.data.images.load("speed_sign.jpg")
    # mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    # if obj.data.materials:
    #     obj.data.materials[0] = mat

    # obj.data.materials.append(mat)

    # ob = bpy.context.active_object

    tex = bpy.data.textures.new(name="Tex_Altura", type="IMAGE")
    img = bpy.data.images.load("/home/abven/CV_Course/badari_p3/EinstienVision/speed_sign.jpg")
    tex.image = img
    tex.extension = 'EXTEND'

    mod = obj.modifiers.new("", 'DISPLACE')
    mod.strength = 0.1
    mod.mid_level = 0
    mod.texture = tex

    # # Make sure we're in Object Mode
    # bpy.ops.object.mode_set(mode='OBJECT')

    # # # Toggle to Edit Mode
    # bpy.ops.object.mode_set(mode='EDIT')

    # # bpy.ops.object.editmode_toggle()
    # bpy.context.space_data.context = 'MATERIAL'
    # bpy.ops.image.open(filepath=image_path,directory='/home/abven/CV_Course/badari_p3/EinstienVision/' , files=[{"name":'speed_sign.jpg' , "name":'speed_sign.jpg'}], relative_path=True, show_multiview=False) # To replace
    # # bpy.ops.image.open(filepath=image_path,directory=directory_path , files=[{"name":image_name , "name":image_name}], relative_path=True, show_multiview=False) # To set image

    # render the scene and save the image
    # get camera and render
    # camera = bpy.data.objects['Camera']
    # camera.location = (0, -5, 1.5)
    # camera.rotation_euler = (1.57, 0, 0)

    # bpy.context.scene.render.filepath = 'output.jpg'
    # bpy.ops.render.render(write_still=True)
    # # Save the scene as a .blend file
    bpy.ops.wm.save_as_mainfile(filepath='scene.blend')


def arg_parser():
    parser = argparse.ArgumentParser(description='Predict on an image')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--obj_name', type=str, help='Name of the object')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    addTexture(args.image_path, args.obj_name)
    # BlenderSceneGenerator.generateScene()