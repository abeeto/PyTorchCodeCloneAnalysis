if "bpy" in locals():
    import importlib
    if "XmlToOgreSkeletonTests" in locals():
        importlib.reload(Tests.XmlToOgreSkeletonTests)
        print("found xmltoorgretests")
    else:
        print("did not find xmltoogretests")
    if "OgreBoneTreeTests" in locals():
        importlib.reload(Tests.OgreBoneTreeTests)

import bpy
import unittest
import mathutils
from mathutils import Vector, Matrix
import xml.etree.ElementTree as ET
from io_Torchlight.Components.OgreBoneTreeProvider import OgreBoneTreeProvider
from io_Torchlight.DTO.OgreSkeleton import OgreSkeletonFromXml

bl_info = {
    "name": "Torchlight2 Ogre3d format",
    "version": (0, 1),
    "author": "Rob James",
    "blender": (2, 78, 0),
    "description": "Import/Export Ogre3d based armatures and animations for Torchlight2",
    "location": "File > Import-Export",
    "description": ("Import-Export Torchlight 2 Skeleton, Import armature, "
                    " and animations"),
    "warning": "",
    "wiki_url": (""),
    "tracker_url": "",
    "support": 'OFFICIAL',
    "category": "Import-Export"}

TO_BLE_MATRIX = mathutils.Matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

TO_ZERO_MATRIX = mathutils.Matrix([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]])

def convertBranch(branch, amt, parent, prot):
    leaf = branch.OgreBone
    bone = amt.edit_bones.new(leaf.Name)
    loc = Vector((leaf.Position.X, leaf.Position.Y, leaf.Position.Z))
    axis = Vector((leaf.Rotation.Axis.X, leaf.Rotation.Axis.Y, leaf.Rotation.Axis.Z))
    rotmat = mathutils.Matrix.Rotation(leaf.Rotation.Angle, 4, axis).inverted()
    #mat_trans = mathutils.Matrix.Translation(loc)
    #mat = mat_trans * rotmat    

    if parent:
        pLoc = parent.head
        amt.edit_bones[leaf.Name].parent = amt.edit_bones[ parent.name ]
    else:
        pLoc = Vector((0,0,0))
    
    #fLoc = ((loc * rotmat) + pLoc) * TO_BLE_MATRIX
    #fLoc = ((loc * rotmat) * TO_BLE_MATRIX) - pLoc
    newpos = prot.transposed() * loc
    
    fLoc = pLoc + newpos
    
    bone.head = fLoc
    #bone.tail = Vector((0,0,1))
    bone.tail = bone.head + (Vector((0,.2,0)) * rotmat)
    
    for childBranch in branch.Children:
        convertBranch(childBranch, amt, bone, rotmat)

if __name__ == '__main__':
    testFile = 'c:\\tmp\\Vanquisher.skeleton.xml'
    xml = ET.parse(testFile)
    xmlRoot = xml.getroot()
    ogreSkeleton = OgreSkeletonFromXml(xmlRoot)
    provider = OgreBoneTreeProvider(ogreSkeleton)
    tree = provider.Fetch()
    
    origin = Vector((0,0,0))
    
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    ob = bpy.context.object
    ob.show_x_ray = True
    ob.name = 'TestRig'
    ob.draw_type = 'WIRE'
    amt = ob.data
    amt.name = 'root'+'Amt'
    amt.show_axes = True
 
    # Create bones
    bpy.ops.object.mode_set(mode='EDIT')
    leaf = tree.OgreBone

    if False:
        bone = amt.edit_bones.new(leaf.Name)
        loc = Vector((leaf.Position.X, leaf.Position.Y, leaf.Position.Z))# * TO_BLE_MATRIX
        bone.head = loc
        axis = Vector((leaf.Rotation.Axis.X, leaf.Rotation.Axis.Y, leaf.Rotation.Axis.Z))
        rotmat = mathutils.Matrix.Rotation(leaf.Rotation.Angle, 4, axis)
        
        childLeaf = tree.Children[0].OgreBone
        cLoc = Vector((childLeaf.Position.X, childLeaf.Position.Y, childLeaf.Position.Z))
        bone.tail = rotmat * cLoc
        
        childBone = amt.edit_bones.new(childLeaf.Name)
        childBone.head = bone.tail

        childAxis = Vector((childLeaf.Rotation.Axis.X, childLeaf.Rotation.Axis.Y, childLeaf.Rotation.Axis.Z))
        childRotMat = mathutils.Matrix.Rotation(childLeaf.Rotation.Angle, 4, childAxis)
        
        childLeaf2 = tree.Children[0].Children[0].OgreBone
        cLoc2 = Vector((childLeaf2.Position.X, childLeaf2.Position.Y, childLeaf2.Position.Z))
        childBone.tail = childRotMat * cLoc2 + childBone.head
        
        
        childBone2 = amt.edit_bones.new(childLeaf2.Name)
        childBone2.head = childBone.tail
        childAxis2 = Vector((childLeaf2.Rotation.Axis.X, childLeaf2.Rotation.Axis.Y, childLeaf2.Rotation.Axis.Z))
        childRotMat2 = mathutils.Matrix.Rotation(childLeaf2.Rotation.Angle, 4, childAxis2)
        childBone2.tail = childRotMat2 * Vector((0,0,1)) + childBone2.head
    else:
        convertBranch(tree, amt, False, TO_ZERO_MATRIX)
