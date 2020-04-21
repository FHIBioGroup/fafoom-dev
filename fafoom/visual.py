#!/usr/bin/python
import os, sys, re
defaults_file = '/home/maksimov/scripts/pymol/defaults.pml'
path = os.getcwd()
open_file = sys.argv[1]
name_of_file = open_file.split('/')[-1].split('.')[0]
format_of_file = open_file.split('/')[-1].split('.')[1]

def draw_picture(open_file, image_write = 'no'):
    name_of_file = open_file.split('/')[-1].split('.')[0]
    format_of_file = open_file.split('/')[-1].split('.')[1]
    if format_of_file == 'in':
        os.system('babel -ifhiaims %s.in -oxyz %s.xyz' % (open_file[:-3], open_file[:-3]))
    print os.getcwd()    
    open_file = os.path.join(os.path.dirname(open_file), '{}.xyz'.format(name_of_file))
    with open(defaults_file, 'w') as defaults:
        defaults.write('loadall %s \n' % (open_file))
        defaults.write('bg white\n')
        defaults.write('set auto_zoom, off\n')
        defaults.write('set specular, off\n')
        defaults.write('set sphere_quality, 2\n')
        defaults.write('\n')
        defaults.write('\n')
        defaults.write('set orthoscopic, on\n')
        defaults.write('set depth_cue, 0\n')
        defaults.write('set two_sided_lighting, 1\n')
        defaults.write('\n')
        defaults.write('color grey85, elem c\n')
        defaults.write('color grey50, elem h\n')
        defaults.write('color salmon, elem o\n')
        defaults.write('color skyblue, elem n\n')
        defaults.write('color orange, elem cu\n')
        defaults.write('color gold, elem au\n')
        defaults.write('\n')
        defaults.write('select hc, elem h+c\n')
        defaults.write('select ho, elem h+o\n')
        defaults.write('select hn, elem h+n\n')
        defaults.write('select hcu, elem h+cu\n')
        defaults.write('select hau, elem h+au\n')
        defaults.write('\n')
        defaults.write('selec cc, elem c+c\n')
        defaults.write('preset.ball_and_stick(selection=\'all\', mode=1)\n')
        defaults.write('set sphere_transparency=0.0, elem h\n')
        defaults.write('set sphere_scale, 0.2, elem h\n')
        defaults.write('set_bond stick_color, grey75, hc, ho, hn, hcu, hau\n')
        defaults.write('set_bond stick_transparency, 0.00, hc, ho, hn, hcu, hau\n')
        defaults.write('set_bond stick_transparency, 0.00, ho\n')
        defaults.write('set_bond stick_transparency, 0.00, hn\n')
        defaults.write('set_bond stick_transparency, 0.00, hcu\n')
        defaults.write('set_bond stick_transparency, 0.00, hau\n')
        defaults.write('\n')
        defaults.write('set_bond stick_radius, 0.07, hc\n')
        defaults.write('set_bond stick_radius, 0.07, ho\n')
        defaults.write('set_bond stick_radius, 0.07, hn\n')
        defaults.write('set_bond stick_radius, 0.07, hcu\n')
        defaults.write('set_bond stick_radius, 0.07, hau\n')
        defaults.write('\n')
        defaults.write('set_bond stick_radius, 0.12, cc\n')
        defaults.write('set_bond stick_transparency, 0, cc\n')
        defaults.write('\n')     
        defaults.write('color salmon, iter0001\n')
        defaults.write('set sphere_transparency=0.75, iter0001\n')
        defaults.write('set_bond stick_color, salmon, iter0001\n')
        defaults.write('set_bond stick_transparency, 0.75, iter0001\n')
        defaults.write('cmd.disable(\'iter0001\')\n')
        defaults.write('cmd.delete(\'hc\')\n')
        defaults.write('cmd.delete(\'ho\')\n')
        defaults.write('cmd.delete(\'hn\')\n')
        defaults.write('cmd.delete(\'hcu\')\n')
        defaults.write('cmd.delete(\'hau\')\n')
        defaults.write('cmd.delete(\'cc\')\n')
        defaults.write('cmd.delete(\'cc\')\n')
        defaults.write('set two_sided_lighting, 1\n')
        defaults.write('set ray_shadows,0\n')
        defaults.write('run /home/maksimov/scripts/pymol/draw_box.py\n')
        defaults.write('\n')
        if image_write == 'yes':
            defaults.write('set ray_opaque_background, off\n')
            defaults.write('set ray_shadows,0\n')
            defaults.write('ray 800 800\n')
            defaults.write('png %s\n' % (os.path.join(os.path.dirname(open_file), 'geometry.png')))
    defaults.close()
    os.system('/usr/bin/pymol -cq -u /home/maksimov/scripts/pymol/defaults.pml')
    
if __name__== '__main__':
    open_file = sys.argv[1]
    draw_picture(os.path.realpath(open_file), image_write = 'yes')
    os.system('/usr/bin/pymol -cq -u /home/maksimov/scripts/pymol/defaults.pml')
    
    
    
