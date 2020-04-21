# fafoom-dev
Continuing development of genetic algorithm (GA) for biomolecules in different environments
﻿# Fafoom - Flexible Algorithm For Optimization Of Molecules 

Fafoom is a tool for sampling the conformational space of organic molecules. Fafoom is intended to work with FHI-aims (Fritz Haber Institute ab initio molecular simulations package).

## News

* the paper "First-principles molecular structure search with a genetic algorithm" is now published in Journal of Chemical Information and Modeling; DOI: 10.1021/acs.jcim.5b00243

* a new branch targeting of providing the stable version of FAFOOM for structure search of isolated molecule

## Requirements

* Functionality of the Fafoom depends on:
  * Python 2.7 
  * Numpy

* first-principles methods:
  * FHI-aims (Fritz Haber Institute ab initio molecular simulations package)

## How to use

##### 1) Clone the fafoom repository

	git clone https://github.com/FHIBioGroup/fafoom-dev.git -b isolated-stable

##### 2) Export the fafoom-dev/fafoom directory to you PYTHONPATH

##### 3) Test in python terminal: 

    import fafoom

##### 4) To execute GA algorithm and prepare your input files and perform structure search: 

    ga.py parameters.txt

Get familiar with the provided manual to learn more about the tool and the parameters. 

## License

Fafoom is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Fafoom is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with fafoom.  If not, see <http://www.gnu.org/licenses/>.


Copyright 2015 Adriana Supady 
