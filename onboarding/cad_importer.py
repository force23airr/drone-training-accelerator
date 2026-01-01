"""
CAD/Model Import System

Import drone specifications from various file formats:
- URDF (Robot Description Format)
- STEP/STP (CAD)
- STL (3D mesh)
- JSON (specification files)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import xml.etree.ElementTree as ET

from onboarding.drone_specification import (
    DroneSpecification,
    AirframeType,
    PropulsionType,
    InertialProperties,
    MotorSpecification,
)


def import_from_json(filepath: str) -> DroneSpecification:
    """
    Import drone specification from JSON file.

    The JSON should follow the DroneSpecification schema.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return DroneSpecification.from_dict(data)


def import_from_urdf(filepath: str) -> DroneSpecification:
    """
    Import drone specification from URDF file.

    Extracts:
    - Mass and inertia from <inertial> tags
    - Dimensions from <visual>/<collision> geometry
    - Joint/link structure for motor positions

    Args:
        filepath: Path to URDF file

    Returns:
        DroneSpecification with extracted parameters
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    spec = DroneSpecification()
    spec.name = root.attrib.get('name', 'Imported URDF Drone')
    spec.urdf_file = filepath

    # Parse links for inertial properties
    total_mass = 0.0
    ixx_total = 0.0
    iyy_total = 0.0
    izz_total = 0.0

    motor_links = []

    for link in root.findall('.//link'):
        link_name = link.attrib.get('name', '')

        # Get inertial properties
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                mass = float(mass_elem.attrib.get('value', 0))
                total_mass += mass

            inertia = inertial.find('inertia')
            if inertia is not None:
                ixx_total += float(inertia.attrib.get('ixx', 0))
                iyy_total += float(inertia.attrib.get('iyy', 0))
                izz_total += float(inertia.attrib.get('izz', 0))

        # Identify motor links
        if 'motor' in link_name.lower() or 'prop' in link_name.lower():
            motor_links.append(link_name)

    # Set mass and inertia
    spec.mass_kg = total_mass
    spec.inertial.mass_total_kg = total_mass
    spec.inertial.ixx = ixx_total
    spec.inertial.iyy = iyy_total
    spec.inertial.izz = izz_total

    # Parse joints for motor positions
    motor_id = 0
    for joint in root.findall('.//joint'):
        child = joint.find('child')
        if child is not None:
            child_link = child.attrib.get('link', '')
            if child_link in motor_links:
                origin = joint.find('origin')
                if origin is not None:
                    xyz = origin.attrib.get('xyz', '0 0 0').split()
                    pos = [float(x) for x in xyz]

                    motor = MotorSpecification(
                        motor_id=motor_id,
                        position_x=pos[0],
                        position_y=pos[1],
                        position_z=pos[2],
                    )
                    spec.motors.append(motor)
                    motor_id += 1

    spec.num_motors = len(spec.motors)

    # Determine airframe type from motor count
    if spec.num_motors == 4:
        spec.airframe_type = AirframeType.QUADCOPTER_X
    elif spec.num_motors == 6:
        spec.airframe_type = AirframeType.HEXACOPTER_X
    elif spec.num_motors == 8:
        spec.airframe_type = AirframeType.OCTOCOPTER_X
    elif spec.num_motors == 1:
        spec.airframe_type = AirframeType.FIXED_WING_CONVENTIONAL

    # Estimate arm length from motor positions
    if spec.motors:
        distances = []
        for motor in spec.motors:
            dist = (motor.position_x**2 + motor.position_y**2) ** 0.5
            distances.append(dist)
        spec.arm_length_m = sum(distances) / len(distances)

    return spec


def import_from_step(filepath: str) -> DroneSpecification:
    """
    Import drone specification from STEP/STP CAD file.

    Requires: pip install OCC or pythonocc-core

    Extracts:
    - Overall dimensions from bounding box
    - Mass (if defined in CAD)
    - Geometry for collision shapes
    """
    spec = DroneSpecification()
    spec.name = Path(filepath).stem
    spec.cad_file = filepath

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add

        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)

        if status == 1:  # Success
            reader.TransferRoots()
            shape = reader.Shape()

            # Get bounding box
            bbox = Bnd_Box()
            brepbndlib_Add(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            spec.length_m = xmax - xmin
            spec.width_m = ymax - ymin
            spec.height_m = zmax - zmin

            # Estimate arm length (half of smaller dimension)
            spec.arm_length_m = min(spec.length_m, spec.width_m) / 2

            print(f"Imported STEP: {spec.length_m:.3f} x {spec.width_m:.3f} x {spec.height_m:.3f} m")
        else:
            print(f"Failed to read STEP file: {filepath}")

    except ImportError:
        print("Warning: pythonocc-core not installed. Install with:")
        print("  conda install -c conda-forge pythonocc-core")
        print("Using placeholder values.")

        # Set placeholder values
        spec.length_m = 0.5
        spec.width_m = 0.5
        spec.height_m = 0.2
        spec.arm_length_m = 0.2

    return spec


def import_from_stl(filepath: str) -> DroneSpecification:
    """
    Import drone specification from STL mesh file.

    Extracts:
    - Dimensions from mesh bounds
    - Volume for mass estimation (if density provided)
    """
    spec = DroneSpecification()
    spec.name = Path(filepath).stem
    spec.mesh_files = [filepath]

    try:
        import numpy as np
        from stl import mesh  # pip install numpy-stl

        # Load STL
        drone_mesh = mesh.Mesh.from_file(filepath)

        # Get bounds
        min_coords = drone_mesh.vectors.min(axis=(0, 1))
        max_coords = drone_mesh.vectors.max(axis=(0, 1))

        spec.length_m = max_coords[0] - min_coords[0]
        spec.width_m = max_coords[1] - min_coords[1]
        spec.height_m = max_coords[2] - min_coords[2]

        # Estimate arm length
        spec.arm_length_m = min(spec.length_m, spec.width_m) / 2

        # Calculate volume (rough estimate)
        volume, _, _ = drone_mesh.get_mass_properties()
        # Note: volume is in mesh units^3

        print(f"Imported STL: {spec.length_m:.3f} x {spec.width_m:.3f} x {spec.height_m:.3f} m")

    except ImportError:
        print("Warning: numpy-stl not installed. Install with:")
        print("  pip install numpy-stl")
        print("Using placeholder values.")

        spec.length_m = 0.5
        spec.width_m = 0.5
        spec.height_m = 0.2
        spec.arm_length_m = 0.2

    return spec


def export_to_urdf(
    spec: DroneSpecification,
    output_path: str,
    mesh_dir: Optional[str] = None,
) -> str:
    """
    Export drone specification to URDF format.

    Args:
        spec: Drone specification to export
        output_path: Path for output URDF file
        mesh_dir: Optional directory containing mesh files

    Returns:
        Path to generated URDF file
    """
    # Create URDF XML
    robot = ET.Element('robot', name=spec.name.replace(' ', '_'))

    # Base link
    base_link = ET.SubElement(robot, 'link', name='base_link')

    # Inertial properties
    inertial = ET.SubElement(base_link, 'inertial')
    ET.SubElement(inertial, 'mass', value=str(spec.mass_kg))
    ET.SubElement(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
    ET.SubElement(inertial, 'inertia',
                  ixx=str(spec.inertial.ixx),
                  iyy=str(spec.inertial.iyy),
                  izz=str(spec.inertial.izz),
                  ixy=str(spec.inertial.ixy),
                  ixz=str(spec.inertial.ixz),
                  iyz=str(spec.inertial.iyz))

    # Visual (simple box representation)
    visual = ET.SubElement(base_link, 'visual')
    vis_origin = ET.SubElement(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
    vis_geom = ET.SubElement(visual, 'geometry')

    if mesh_dir and spec.mesh_files:
        # Use mesh file if available
        mesh_path = f"{mesh_dir}/{spec.mesh_files[0]}"
        ET.SubElement(vis_geom, 'mesh', filename=mesh_path)
    else:
        # Use box primitive
        ET.SubElement(vis_geom, 'box', size=f'{spec.arm_length_m*2} {spec.arm_length_m*2} {spec.height_m}')

    # Collision (same as visual)
    collision = ET.SubElement(base_link, 'collision')
    col_origin = ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
    col_geom = ET.SubElement(collision, 'geometry')
    ET.SubElement(col_geom, 'box', size=f'{spec.arm_length_m*2} {spec.arm_length_m*2} {spec.height_m}')

    # Motor links and joints
    for motor in spec.motors:
        motor_name = f'motor_{motor.motor_id}'

        # Motor link
        motor_link = ET.SubElement(robot, 'link', name=motor_name)
        motor_inertial = ET.SubElement(motor_link, 'inertial')
        ET.SubElement(motor_inertial, 'mass', value='0.05')  # Approximate motor mass
        ET.SubElement(motor_inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(motor_inertial, 'inertia',
                      ixx='0.0001', iyy='0.0001', izz='0.0001',
                      ixy='0', ixz='0', iyz='0')

        # Motor visual (cylinder)
        motor_vis = ET.SubElement(motor_link, 'visual')
        ET.SubElement(motor_vis, 'origin', xyz='0 0 0', rpy='0 0 0')
        motor_vis_geom = ET.SubElement(motor_vis, 'geometry')
        ET.SubElement(motor_vis_geom, 'cylinder', radius='0.02', length='0.03')

        # Joint connecting motor to base
        joint = ET.SubElement(robot, 'joint', name=f'{motor_name}_joint', type='fixed')
        ET.SubElement(joint, 'parent', link='base_link')
        ET.SubElement(joint, 'child', link=motor_name)
        ET.SubElement(joint, 'origin',
                      xyz=f'{motor.position_x} {motor.position_y} {motor.position_z}',
                      rpy='0 0 0')

    # Write URDF file
    tree = ET.ElementTree(robot)
    ET.indent(tree, space='  ')

    with open(output_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)

    print(f"Exported URDF to: {output_path}")
    return output_path


def export_to_json(spec: DroneSpecification, output_path: str) -> str:
    """Export drone specification to JSON file."""
    spec.save(output_path)
    print(f"Exported JSON to: {output_path}")
    return output_path
