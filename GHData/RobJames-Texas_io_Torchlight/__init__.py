# ====================== BEGIN GPL LICENSE BLOCK ======================#
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ======================= END GPL LICENSE BLOCK ========================#

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
from io_Torchlight.Tests import XmlToOgreSkeletonTests
from io_Torchlight.Tests import OgreBoneTreeTests 
from io_Torchlight.Tests.XmlToOgreSkeletonTests import XmlToOgreSkeletonTests
from io_Torchlight.Tests.OgreBoneTreeTests import OgreBoneTreeTests


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


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
