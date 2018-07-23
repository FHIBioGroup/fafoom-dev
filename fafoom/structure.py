#    Copyright 2015 Adriana Supady
#
#    This file is part of fafoom.
#
#   Fafoom is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   Fafoom is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#   along with fafoom.  If not, see <http://www.gnu.org/licenses/>.
''' Handle the molecule and its 3D structures.'''
from __future__ import division
from copy import deepcopy
from get_parameters import (
    create_dof_object,
    get_atoms_and_bonds,
    get_positions,
    template_sdf
)
from genetic_operations import crossover_single_point, crossover_random_points
from pyaims import AimsObject
from pynwchem import NWChemObject
from pyorca import OrcaObject
#from pytest import TESTObject
#from pyff import FFObject
from pyforcefield import FFobject
from deg_of_freedom import Centroid, Protomeric, NumberOfMolecules
from measure import *
import numpy as np


class MoleculeDescription:
    """Create the molecule."""
    newline = "NEWLINE"

    def __init__(self, parameter_file=None, **kwargs):
        """Initialize the molecule. Get the parameters from file (if present)
        or read keyword args. The keyword args overwrite the file values."""
        params = {}
        if parameter_file is not None:
            params = file2dict(parameter_file, ['Molecule'])
            new_names_dict = {'smile': 'smiles',
                              'smart_torsion': 'smarts_torsion',
                              'filter_smart_torsion': 'filter_smarts_torsion',
                              'smart_cistrans': 'smarts_cistrans'}
            for key in new_names_dict:
                try:
                    params[new_names_dict[key]] = params.pop(key)
                except KeyError:
                    pass
        else:
            for key in kwargs.keys():
                if key not in ["template_sdf_string"]:
                    params[str(key)] = kwargs[key]
                else:
                    params[str(key)] = kwargs[key].replace(
                        MoleculeDescription.newline, "\n")

        dict_default = {'rmsd_type': "internal",
                        'rmsd_cutoff_uniq': 0.2,
                        'chiral': False,
                        'optimize_torsion': True,
                        'optimize_cistrans': False,
                        'optimize_centroid': False,
                        'optimize_orientation': False,
                        'optimize_protomeric': False,
                        'sourcedir': 'adds',
                        'sdf_string_template': 'adds/mol.sdf',
                        'constrained_geometry_file': 'adds/geometry.in.constrained',
                        'right_order_to_assign': ['torsion', 'cistrans', 'centroid', 'orientation', 'protomeric'],
                        'volume': (-10, 11, -10, 11, -10, 11),
                        'number_of_protons': 0,
                        'number_of_molecules': 1,
                        'molecules': 'same',
                        'crossover_method': 'single_point'}

        params = set_default(params, dict_default)
        for key in params:
            if not hasattr(self, str(key)):
                setattr(self, str(key), params[key])

    def __repr__(self):
        """Create an unambiguous object representation. The resulting string
        is an one-liner with the newline parameter replacing the original
        '\n' sign in the template sdf_string attribute."""
        repr_list = []
        for att_name in self.__dict__.keys():

            if type(self.__dict__[att_name]) in [str] and \
                            att_name != "template_sdf_string":
                repr_list.append('%s="%s"' %
                                 (att_name, getattr(self, att_name)))

            elif type(self.__dict__[att_name]) in [int, float, bool, list]:

                repr_list.append('%s=%s' %
                                 (att_name, repr(getattr(self, att_name))))
            elif att_name in ["template_sdf_string"]:
                repr_list.append("%s='%s'" % (
                    att_name, getattr(
                        self, att_name).replace("\n",
                                                MoleculeDescription.newline, )))
                # else:
                # print_output("Unknown type of attribute "+str(att_name))
        return "%s(%s)" % (self.__class__.__name__, ', '.join(repr_list))

    def __eq__(self, other):
        """Compare all attribute values of two objects. Returns True if all
        values are identical."""
        for att_name in self.__dict__.keys():
            if getattr(self, att_name) == getattr(other, att_name):
                continue
            else:
                return False
        for att_name in other.__dict__.keys():
            if getattr(other, att_name) == getattr(self, att_name):
                continue
            else:
                return False
        return True

    def analyze_constrained_geometry(self):
        NumOfAtoms = 0
        Periodic = False
        Path = os.path.join(os.getcwd(), self.constrained_geometry_file)
        with open(Path) as constrained:
            lines = constrained.readlines()
            for line in lines:
                if 'lattice_vector' in line:
                    Periodic = True
                if 'atom' in line:
                    NumOfAtoms += 1
        return NumOfAtoms, Periodic, Path

    def UpdateSharedBlacklist(self, blacklist=[], folders=[]):
        other_GAs = []
        shared_blacklist = blacklist
        visited_folders = folders
        parent_directory = ('/').join(os.getcwd().split('/')[:-1])
        for i in os.listdir(parent_directory):
            if 'GARUN' in i and os.path.isdir(os.path.join(parent_directory, i)):
                for calculated_structure in os.listdir(os.path.join(parent_directory, i)):
                    if calculated_structure != 'adds' and os.path.isdir(
                            os.path.join(parent_directory, i, calculated_structure)):
                        other_GAs.append(os.path.join(parent_directory, i, calculated_structure))
        for str_folder in other_GAs:
            print(str_folder)
            if str_folder not in visited_folders:
                if os.path.exists(os.path.join(str_folder, 'geometry_out.sdf')):
                    with open(os.path.join(str_folder, 'geometry_out.sdf')) as sdf_out:
                        str3d = Structure(self)
                        str3d.index = len(shared_blacklist) + 1000
                        str3d.generate_structure()
                        out_string = sdf_out.read()
                        for dof in str3d.dof:
                            dof.update_values(out_string)
                        str3d.send_to_blacklist(shared_blacklist)
                    with open(os.path.join(str_folder, 'geometry_in.sdf')) as sdf_out:
                        str3d = Structure(self)
                        str3d.index = len(shared_blacklist) + 10000
                        str3d.generate_structure()
                        in_string = sdf_out.read()
                        for dof in str3d.dof:
                            dof.update_values(in_string)
                        str3d.send_to_blacklist(shared_blacklist)
                    visited_folders.append(str_folder)
        return shared_blacklist, visited_folders

    def UpdateBlacklist(self, blacklist=[], folders=[]):
        other_GAs = []
        shared_blacklist = blacklist
        visited_folders = folders
        parent_directory = ('/').join(os.getcwd().split('/')[:-1])
        """Check all the folders"""
        for i in os.listdir(parent_directory):
            if 'GARUN' in i:
                for calculated_structure in os.listdir(os.path.join(parent_directory, i)):
                    if calculated_structure != 'adds' and os.path.isdir(
                            os.path.join(parent_directory, i, calculated_structure)):
                        other_GAs.append(os.path.join(parent_directory, i, calculated_structure))
        """Visit new folders"""
        for str_folder in other_GAs:
            if str_folder not in visited_folders:
                if os.path.exists(os.path.join(str_folder, 'geometry_out.sdf')):
                    with open(os.path.join(str_folder, 'geometry_out.sdf')) as sdf_out:
                        str3d = Structure(self)
                        str3d.index = len(shared_blacklist)
                        str3d.generate_structure()
                        out_string = sdf_out.read()
                        for dof in str3d.dof:
                            dof.update_values(out_string)
                        str3d.send_to_blacklist(shared_blacklist)
                    with open(os.path.join(str_folder, 'geometry_in.sdf')) as sdf_out:
                        str3d = Structure(self)
                        str3d.index = len(shared_blacklist)
                        str3d.generate_structure()
                        in_string = sdf_out.read()
                        for dof in str3d.dof:
                            dof.update_values(in_string)
                        str3d.send_to_blacklist(shared_blacklist)
                    visited_folders.append(str_folder)
        return shared_blacklist, visited_folders



    def create_template_sdf(self):
        """Assign new attribute (template_sdf_string) to the object."""
        self.template_sdf_string = template_sdf(self.sdf_string_template)  # Need to revise, looks very confusing.

    def get_parameters(self):
        with open(os.path.join(os.getcwd(), self.sdf_string_template), 'r') as sdf_file:
            self.sdf_string_template = sdf_file.read()

        """Assign permanent attributes (number of atoms, number of bonds and
        degrees of freedom related attributes) to the object."""
        self.atoms, self.bonds = get_atoms_and_bonds(self.sdf_string_template)
        self_copy = deepcopy(self)
        dof_names = []
        for attr, value in self_copy.__dict__.iteritems():
            if str(attr).split('_')[0] == "optimize" and value:
                type_of_dof = str(attr).split('_')[1]
                linked_attr = {}
                for attr, value in self_copy.__dict__.iteritems():
                    if type_of_dof in str(attr).split('_'):
                        linked_attr[attr] = value
                pos = get_positions(type_of_dof, self.sdf_string_template, **linked_attr)
                if len(pos) != 0:
                    setattr(self, type_of_dof, pos)
                    dof_names.append(type_of_dof)
                else:
                    print_output("The degree to optimize: " + str(type_of_dof) +
                                 " hasn't been found.")

        geom_file = os.path.join(os.getcwd(), self.constrained_geometry_file)
        if os.path.isfile(geom_file):
            if len(aims2xyz(geom_file)) == 0:
                if 'centroid' in dof_names:
                    dof_names.remove('centroid')
                if 'orientation' in dof_names:
                    dof_names.remove('orientation')
        else:
            if 'centroid' in dof_names:
                dof_names.remove('centroid')
            if 'orientation' in dof_names:
                dof_names.remove('orientation')
            open(geom_file, 'w').close()

        updated_order = []
        for i in self.right_order_to_assign:
            if i in dof_names:
                updated_order.append(i)
        setattr(self, "dof_names", updated_order)

        Centroid.range_x = range(self.volume[0], self.volume[1], 1)  # Limitation for Centroid
        Centroid.range_y = range(self.volume[2], self.volume[3], 1)  # Limitation for Centroid
        Centroid.range_z = range(self.volume[4], self.volume[5], 1)  # Limitation for Centroid
        Centroid.values_options = [Centroid.range_x, Centroid.range_y, Centroid.range_z]

        # Routines for Protomeric values:
        if 'protomeric' in dof_names:
            for i in self.list_of_protomeric:
                Protomeric.maximum_of_protons.append(i[-1])
            max_num_of_protons = sum(Protomeric.maximum_of_protons)
            Protomeric.number_of_protons = self.number_of_protons
            if max_num_of_protons - self.number_of_protons > len(self.list_of_protomeric):
                num_of_zeros = len(self.list_of_protomeric)
            else:
                num_of_zeros = max_num_of_protons - self.number_of_protons
            Protomeric.values_options = np.lib.pad(
                np.ones(len(self.list_of_protomeric) - (max_num_of_protons - self.number_of_protons)),
                (0, num_of_zeros),
                'constant', constant_values=(0))
        # Routines for number of molecules:
        NumberOfMolecules.numofmol = self.number_of_molecules


class Structure:
    """Create 3D structures."""
    index = 0
    newline = "NEWLINE"

    def __init__(self, arg=None, **kwargs):
        """Initialize the 3D structure: (1) from MoleculeDescription class
        object or from (2) from previously created object of the Structure
        class. Any present and valid keyword args overwrite the old values.
        Warning: there may be more attributes in the (2) case."""
        if isinstance(arg, MoleculeDescription):

            self.mol_info = arg
            Structure.index += 1
            self.index = Structure.index
            dof = []
            for i in self.mol_info.dof_names:
                new_obj = create_dof_object(str(i), getattr(self.mol_info, i))
                dof.append(new_obj)
            setattr(self, "dof", dof)

        elif isinstance(arg, Structure):

            self.mol_info = arg.mol_info
            Structure.index += 1
            self.index = Structure.index
            for att_name in arg.__dict__.keys():

                if att_name not in ["mol_info", "index"]:
                    setattr(self, str(att_name),
                            deepcopy(getattr(arg, str(att_name))))

        else:
            print_output("Initialization can't be performed. Check the input")

        for key in kwargs.keys():
            if key != "index":
                if hasattr(self, str(key)):
                    print_output("Overwriting the value for keyword " + str(key))
                    print_output("Old value: " + str(getattr(self, str(key))) +
                                 ", new value: " + str(kwargs[key]))
                if key in ["sdf_string", "initial_sdf_string"]:
                    setattr(self, str(key),
                            kwargs[key].replace(Structure.newline, "\n"))

                elif key.split('_')[0] in self.mol_info.dof_names:
                    for dof in self.dof:
                        if key.split('_')[0] == dof.type:
                            if key.split('_')[1] == 'initial':
                                setattr(dof, 'initial_values', kwargs[key])
                            if key.split('_')[1] == 'values':
                                setattr(dof, 'values', kwargs[key])
                else:
                    setattr(self, str(key), kwargs[key])

    def __repr__(self):
        """Create an unambiguous object representation. The resulting string
        is an one-liner with the newline parameter replacing the original
        '\n' sign in the sdf_string and initial_sdf_string attribute."""
        repr_list = []
        for att_name in self.__dict__.keys():

            if att_name in ["sdf_string", "initial_sdf_string"]:
                repr_list.append("%s='%s'" % (
                    att_name, getattr(
                        self, att_name).replace("\n",
                                                Structure.newline)))
            else:
                if type(self.__dict__[att_name]) in [str]:
                    repr_list.append('%s=%s' % (
                        att_name, repr(getattr(self, att_name))))
                elif type(self.__dict__[att_name]) in [int, float, bool]:
                    repr_list.append('%s=%s' % (
                        att_name, repr(getattr(self, att_name))))
                elif att_name == 'dof':
                    for dof in self.dof:
                        repr_list.append('%s_%s=%s' % (
                            dof.type, "values", repr(dof.values)))
                        try:
                            repr_list.append('%s_%s=%s' % (
                                dof.type, "initial_values",
                                repr(dof.initial_values)))
                        except:
                            pass
                elif att_name == 'mol_info':
                    pass
                    # else:
                    # print_output("Unknown type of attribute "+str(att_name))

        return "%s(mol, %s)" % (self.__class__.__name__, ', '.join(repr_list))

    def __str__(self):
        """Return the object index."""
        return "%s %d" % (self.__class__.__name__, self.index)

    def __float__(self):
        """Return the object energy."""
        return float(self.energy)

    def generate_structure(self, values={}):
        """Generate a 3D structures. If no values are passed, a random
        structure will be generated (weights, associated with the degrees of
        freedom, will be taken into account)."""
        new_string = deepcopy(self.mol_info.template_sdf_string)
        for dof in self.dof:
            if dof.type in values.keys():
                new_string = dof.apply_on_string(new_string, values[dof.type])
            else:
                if hasattr(self.mol_info, "weights_" + str(dof.type)):
                    weights = getattr(self.mol_info, "weights_" + str(dof.type))
                    dof.get_weighted_values(weights)
                else:
                    dof.get_random_values()
                new_string = dof.apply_on_string(new_string)
        self.sdf_string = new_string
        for dof in self.dof:
            dof.update_values(self.sdf_string)

    def is_geometry_valid(self, flag):
        """Return True if the geometry is valid."""
        check = check_geo_sdf(self.sdf_string, flag=flag)
        return check

    def is_geometry_valid_after_crossover(self):
        """Return True if the geometry is valid."""
        check = check_geo_sdf_after_crossover(self.sdf_string)
        return check

    def __eq__(self, other):
        """Decide, if the objects are equal based on the rms values.

        Returns:
            True, if the objects are 'similar'
        Raises:
            ValueError: if the rmsd type differs between the objects
            ValueErrof: if the rmsd type is unknown (supported options are
            'cartesian' and 'internal_coord')
        """

        if self.mol_info.rmsd_type != other.mol_info.rmsd_type:
            raise ValueError("The type of the rmsd differ for the objects")
        if self.mol_info.rmsd_type not in ['cartesian', 'internal_coord']:
            raise ValueError("Unknown type of rmsd.")

        obj1, obj2 = self, other

        """ To VERIFY """
        # if hasattr(self, "initial_sdf_string"):
            # obj1, obj2 = obj2, obj1
        # if hasattr(obj1, "initial_sdf_string"):
        #     raise Exception("Both structures are already relaxed.")

        if obj1.mol_info.rmsd_type == 'cartesian':
            linked_strings = {}

            if hasattr(obj2, "initial_sdf_string"):
                n_str = str(obj2.initial_sdf_string)
                linked_strings[n_str] = obj2.initial_sdf_string

            if not obj1.mol_info.chiral:
                n_str = str(mirror_sdf(obj2.sdf_string))
                linked_strings[n_str] = mirror_sdf(obj2.sdf_string)
                if hasattr(obj2, "initial_sdf_string"):
                    n_str = str(mirror_sdf(obj2.initial_sdf_string))
                    linked_strings[n_str] = mirror_sdf(obj2.initial_sdf_string)

            bestrms = lowest_cartesian(obj1.sdf_string, obj2.sdf_string,
                                       **linked_strings)

            if bestrms > obj1.mol_info.rmsd_cutoff_uniq:
                return False
            else:
                return True

        if obj1.mol_info.rmsd_type == 'internal_coord':
            all_bool = []
            for dof1, dof2 in zip(obj1.dof, obj2.dof):
                all_bool.append(dof1.is_equal(dof2,
                                              obj1.mol_info.rmsd_cutoff_uniq,
                                              obj1.mol_info.chiral))

            if False in all_bool:
                return False
            else:
                return True

    def __cmp__(self, other):
        """Compare two object basing on their energy values."""
        return cmp(self.energy, other.energy)

    def send_to_blacklist(self, array):
        """Append the structure to dedicated array.

        Args:
           array: the array to append to
        Raise:
            NameError: if the array not defined
        """

        array.append(self)

    def send_to_new_blacklist(self, array):
        """Append the structure to dedicated array.

        Args:
           array: the array to append to
        Raise:
            NameError: if the array not defined
        """

        def produce_header(sdf_string):
            all_lines = sdf_string.splitlines()
            all_lines[0] = ' FAFOOM powered'
            all_lines[1] = ' Index = {}'.format(self.index)
            all_lines[2] = ' Energy = {}'.format(float(self))
            return '\n'.join(all_lines)

        updated_string = produce_header(self.sdf_string)
        structure_for_blacklist = '{}\n$$$$\n'.format(updated_string)
        array.append(structure_for_blacklist)

    def send_to_shared_blacklist(self, array):
        """Append the structure to dedicated array.

        Args:
           array: the array to append to
        Raise:
            NameError: if the array not defined
        """

        def produce_header(sdf_string):
            all_lines = sdf_string.splitlines()
            all_lines[0] = ' FAFOOM powered'
            all_lines[1] = ' Index = {}'.format(self.index)
            all_lines[2] = ' Energy = {}'.format(0)
            return '\n'.join(all_lines)

        updated_string = produce_header(self.sdf_string)
        structure_for_blacklist = '{}\n$$$$\n'.format(updated_string)
        array.append(structure_for_blacklist)

    def perform_random(self, sourcedir, dirname):
        """Stores random structures without futher calculations"""
        aims_object = AimsObject(sourcedir)
        aims_object.generate_input(self.sdf_string)
        aims_object.build_storage(dirname)
        self.energy = np.random.rand()
        self.initial_sdf_string = self.sdf_string

    def perform_random_test(self, sourcedir, dirname):
        """Stores random structures without futher calculations"""
        test_object = TESTObject(sourcedir)
        self.initial_sdf_string = self.sdf_string
        test_object.generate_input(self.sdf_string)
        self.generate_structure()
        test_object.generate_output(self.sdf_string)
        test_object.build_storage(dirname)
        self.energy = np.random.rand()
        for dof in self.dof:
            dof.update_values(self.sdf_string)


    def perform_aims(self, sourcedir, execution_string, dirname):
        """Generate the FHI-aims input, run FHI-aims, store the output, assign
        new attributes and update attribute values."""

        aims_object = AimsObject(sourcedir)
        aims_object.generate_input(self.sdf_string)
        aims_object.build_storage(dirname)
        success = aims_object.run_aims(execution_string)
        if success:
            aims_object.clean_and_store()
            self.energy = aims_object.get_energy()
            self.initial_sdf_string = self.sdf_string
            string2file(self.initial_sdf_string, 'geometry_in.sdf')
            self.sdf_string = aims2sdf(aims_object.get_aims_string_opt(),
                                       self.mol_info.template_sdf_string)
            string2file(self.sdf_string, 'geometry_out.sdf')

            for dof in self.dof:
                setattr(dof, "initial_values", dof.values)
                dof.update_values(self.sdf_string)
        else:
            print_output("The FHI-aims relaxation failed")

    def perform_aims_single_point(self, sourcedir, execution_string, dirname):
        """Generate the FHI-aims input, run FHI-aims, store the output, assign
        new attributes and update attribute values."""

        aims_object = AimsObject(sourcedir)
        aims_object.generate_input_single_point(self.sdf_string)
        aims_object.build_storage(dirname)
        success = aims_object.run_aims(execution_string)
        aims_object.clean_and_store()
        for dof in self.dof:
            setattr(dof, "initial_values", dof.values)
            dof.update_values(self.sdf_string)
            # ~ if success:

            # ~ self.energy = aims_object.get_energy()
            # ~ self.initial_sdf_string = self.sdf_string
            # ~ self.sdf_string = aims2sdf(aims_object.get_aims_string_opt(),
            # ~ self.mol_info.template_sdf_string)
            # ~ else:
            # ~ print_output("The FHI-aims relaxation failed")

    def perform_FF(self, sourcedir, execution_string, dirname):
        FF_object = FFobject(sourcedir)
        FF_object.generate_input(self.sdf_string)
        FF_object.build_storage(dirname)
        FF_object.run_FF(execution_string)
        self.energy = FF_object.get_energy()
        self.initial_sdf_string = self.sdf_string
        with open(os.path.join(os.getcwd(), dirname, 'geometry_in.sdf'), 'w') as input_sdf:
            input_sdf.write(self.initial_sdf_string)
        self.sdf_string = xyz2sdf(FF_object.get_FF_string_opt(), self.mol_info.template_sdf_string)
        with open(os.path.join(os.getcwd(), dirname, 'geometry_out.sdf'), 'w') as resulted_sdf:
            resulted_sdf.write(self.sdf_string)
        for dof in self.dof:
            setattr(dof, "initial_values", dof.values)
            dof.update_values(self.sdf_string)

    def perform_nwchem(self, functional, basis_set, execution_string):
        """Generate the NWChem input, run NWChem, assign new attributes and
        update attribute values."""
        nwchem_object = NWChemObject(functional, basis_set)
        nwchem_object.clean()
        nwchem_object.generate_input(self.sdf_string)
        nwchem_object.run_nwchem(execution_string)
        nwchem_object.clean()
        self.energy = nwchem_object.get_energy()
        self.initial_sdf_string = self.sdf_string
        self.sdf_string = xyz2sdf(nwchem_object.get_xyz_string_opt(),
                                  self.mol_info.template_sdf_string)

        for dof in self.dof:
            setattr(dof, "initial_values", dof.values)
            dof.update_values(self.sdf_string)

    def perform_orca(self, commandline, memory, execution_string, **kwargs):
        """Generate the orca input, run orca, assign new attributes and
        update attribute values."""
        orca_object = OrcaObject(commandline, memory, **kwargs)
        orca_object.clean()
        orca_object.generate_input(self.sdf_string)
        orca_object.run_orca(execution_string)
        orca_object.clean()
        self.energy = orca_object.get_energy()
        self.initial_sdf_string = self.sdf_string
        self.sdf_string = xyz2sdf(orca_object.get_xyz_string_opt(),
                                  self.mol_info.template_sdf_string)

        for dof in self.dof:
            setattr(dof, "initial_values", dof.values)
            dof.update_values(self.sdf_string)

    def perform_ff(self, force_field, **kwargs):
        """Generate the force-field input, run force=field calculation, assign
        new attributes and update attribute values."""
        ff_object = FFObject(force_field, **kwargs)
        ff_object.run_ff(self.sdf_string)
        self.energy = ff_object.get_energy()
        self.initial_sdf_string = self.sdf_string
        self.sdf_string = ff_object.get_sdf_string_opt()
        ff_object.save_to_file()
        for dof in self.dof:
            setattr(dof, "initial_values", dof.values)
            dof.update_values(self.sdf_string)

    def crossover(self, other, method='random_points'):
        """Perform the crossover."""
        child1 = Structure(self.mol_info)
        child2 = Structure(self.mol_info)
        for dof_par1, dof_par2, dof_child1, dof_child2 in zip(self.dof, other.dof,
                                                              child1.dof, child2.dof):
            """ Crossing over for Centre of mass and for Orientation of the
            main axis of inertia is performed as swapping of the whole vectors
            without dividing them into parts."""
            if dof_par1.type == dof_par2.type:
                if dof_par1.type == 'orientation' or dof_par1.type == 'centroid' or dof_par1.type == 'protomeric':
                    a, b = getattr(dof_par1, "values"), getattr(dof_par2, "values")
                    setattr(dof_child1, "values", b)
                    setattr(dof_child2, "values", a)
                else:
                    if method == 'random_points':
                        a, b = crossover_random_points(getattr(dof_par1, "values"),
                                                       getattr(dof_par2, "values"))
                        setattr(dof_child1, "values", a)
                        setattr(dof_child2, "values", b)
                    if method == 'single_point':
                        a, b = crossover_single_point(getattr(dof_par1, "values"),
                                                      getattr(dof_par2, "values"))
                        setattr(dof_child1, "values", a)
                        setattr(dof_child2, "values", b)
        for child in child1, child2:
            new_string = deepcopy(child.mol_info.template_sdf_string)
            for dof in child.dof:
                new_string = dof.apply_on_string(new_string, values_to_set=dof.values)
            child.sdf_string = new_string
            for dof in child.dof:
                dof.update_values(child.sdf_string)
        return child1

    def mutate(self, **kwargs):
        """ Perform mutation """
        def call_mut(dof, max_mutations=None, weights=None):
            if max_mutations is not None:
                if hasattr(self.mol_info, "weights_" + str(dof.type)):
                    weights = getattr(self.mol_info, "weights_" + str(dof.type))
                    dof.mutate_values(max_mutations, weights)
                else:
                    dof.mutate_values(max_mutations=max_mutations)
            else:
                if hasattr(self.mol_info, "weights_" + str(dof.type)):
                    weights = getattr(self.mol_info, "weights_" + str(dof.type))
                    dof.mutate_values(weights=weights)
                else:
                    dof.mutate_values()
        for dof in self.dof:
            if 'prob_for_mut_' + str(dof.type) in kwargs:
                if np.random.rand() < kwargs['prob_for_mut_' + str(dof.type)]:
                    if 'max_mutations_' + str(dof.type) in kwargs:
                        call_mut(dof, kwargs['max_mutations_' + str(dof.type)])
                    else:
                        call_mut(dof)
            else:
                if 'max_mutations_' + str(dof.type) in kwargs:
                    call_mut(dof, kwargs['max_mutations_' + str(dof.type)])
                else:
                    call_mut(dof)
        template = Structure(self.mol_info)
        new_string = template.mol_info.template_sdf_string      # Maybe not necessary
        for dof in self.dof:
            new_string = dof.apply_on_string(new_string, dof.values)
        self.sdf_string = new_string
        for dof in self.dof:
            dof.update_values(self.sdf_string)

    def put_to_origin(self):
        new_coords = align_to_axes(self.sdf_string, 0, 1)
        COM = get_centre_of_mass_from_sdf(self.sdf_string)
        coordinates_at_origin = new_coords[:, :3] - COM
        updated_sdf = update_coords_sdf(self.sdf_string, coordinates_at_origin)
        self.sdf_string = updated_sdf
        for dof in self.dof:
            dof.update_values(self.sdf_string)

    def adjust_position(self):
        mol = [float(i) for i in np.array(sdf2xyz(self.sdf_string))[:, 3]]
        surr = [float(i) for i in aims2xyz(self.mol_info.constrained_geometry_file)[:, 3]]
        z_min = min(mol)
        z_max = max(surr)
        atom_min = sdf2xyz(self.sdf_string)[mol.index(z_min)][0]
        atom_max = aims2xyz(self.mol_info.constrained_geometry_file)[surr.index(z_max)][0]
        dist = (VDW_radii[atom_min] + VDW_radii[atom_max]) * 0.5  # + VDW_radii[atom_max]
        values_old = centroid_measure(self.sdf_string)
        values_new = np.array([(self.mol_info.volume[0] + self.mol_info.volume[1]) / 2,
                               (self.mol_info.volume[2] + self.mol_info.volume[3]) / 2,
                               values_old[2] - (z_min - z_max) + dist])
        new_string = centroid_set(self.sdf_string, values_new)
        self.sdf_string = new_string
        for dof in self.dof:
            dof.update_values(self.sdf_string)

    def adjust_position_ion(self):
        """ Adjust position of the molecule with respect to the single Atom  placed in the origin """
        def cart2sph(x, y, z):      # Cartesian to Spherical coordinates
            hxy = np.hypot(x, y)
            r = np.hypot(hxy, z)
            az = np.arctan2(y, x)
            el = np.arctan2(hxy, z)
            return r, az, el

        def sph2cart(r, az, el):    # Spherical to Cartesian coordinates
            x = r * np.sin(el) * np.cos(az)
            y = r * np.sin(el) * np.sin(az)
            z = r * np.cos(el)
            return x, y, z

        constrained_geom_file = self.mol_info.constrained_geometry_file
        # Obtain vdW radii from the file
        constrained = aims2xyz_vdw(constrained_geom_file)[0][0]
        # Obtain vdW radiis and coordinates of the molecule:
        temp = sdf2xyz_list(self.sdf_string)
        # Obtain distances of all atoms to the origin:
        mol_distances = [np.linalg.norm(np.array([float(i[0]), float(i[1]), float(i[2])])) for i in
               np.array(sdf2xyz(self.sdf_string))[:, 1:]]
        # Calculate centre of mass of the molecule:
        com = get_centre_of_mass_from_sdf(self.sdf_string)
        # Calculate the adjustment that should be done in spherical coordinates:
        # Half of the sum of molecule and ion vdW distances (distance to be set):
        D = sum([constrained + temp[mol_distances.index(min(mol_distances))][0]])*0.5
        # Coordinates of the atom nearest to the ion:
        coor1 = np.array(temp[mol_distances.index(min(mol_distances))][1:])
        # Coordinates of the atom at distance that had to be set after adjusting
        # of its coordinates in spherical coordinates
        coor2 = np.array(sph2cart(D,
                                       cart2sph(coor1[0], coor1[1], coor1[2])[1],
                                       cart2sph(coor1[0], coor1[1], coor1[2])[2]))
        # Vector of displacement that will be applied to the COM of the molecule:
        displacement = coor1 - coor2
        # Apply changes to the molecule:
        new_string = centroid_set(self.sdf_string, com - displacement)
        self.sdf_string = new_string
        for dof in self.dof:
            if dof.name == 'Centroid':
                dof.update_values(self.sdf_string)

    def adjust_xy(self, lat_vectors):
        x = lat_vectors[0][0]*0.5
        y = lat_vectors[1][1]*0.5
        com = get_centre_of_mass_from_sdf(self.sdf_string)
        values_new = np.array([x, y, com[2]])
        new_string = centroid_set(self.sdf_string, values_new)
        self.sdf_string = new_string
        for dof in self.dof:
            if dof.name == 'Centroid':
                dof.update_values(self.sdf_string)

    def prepare_for_calculation(self, NumOfAtoms_sur, Periodic_sur, Path_sur):
        def extract_lattice_vectors(Path_sur):
            vectors = []
            with open(Path_sur) as constrained:
                lines = constrained.readlines()
                for line in lines:
                    if 'lattice_vector' in line:
                        vectors.append([float(x) for x in line.split()[1:]])
            return vectors
        if NumOfAtoms_sur == 0:           # If the geometry.in.constrained file is empty:
            self.put_to_origin()          # Put the molecule in the origin for convenience.
        elif NumOfAtoms_sur >= 1 and not Periodic_sur:   # Single Atom in the origin
            # Always perform adjustment of the position with respect ot single Atom
            self.adjust_position_ion()
        elif NumOfAtoms_sur >= 1 and Periodic_sur:
            # Adjust height of the molecule
            self.adjust_position()
            # FOR NOW put molecule at centre of slab
            self.adjust_xy(extract_lattice_vectors(Path_sur))

