MiniZinc models and data files
accompanies the Constraints journal article "A New Constraint Programming Model and Solving for the Cyclic Hoist Scheduling Problem", and the CPAIOR 2020 abstract of the same title

copyright (c) 2020 M. Wallace and N. Yorke-Smith
contact: n.yorke-smith@tudelft.nl
version: September 2020
released under CC BY-NC-SA license (https://creativecommons.org/licenses/by-nc-sa/4.0/)


The contents is the following:

*description*

README.txt (this file)

*problem instances*

BO1 -- BO1.dzn
P&U -- PU.dzn

*problem parameter instances*

in folder dzn with names as follows:

Multiplier-Hoists-Capacity -- M-H-C.dzn

*MiniZinc models*

Che [5]                  -- che-hoist.mzn
Riera & Yorke-Smith [17] -- riera-hoist_nojoblimit.mzn
CECnH (Constraints)      -- wallace-hoist_cpaior20-submission.mzn

*solutions*

output on the PU instances of Table 1 and 2 -- solutions-PU.txt

note: run on a different machine than the one used for the Constraint article


To run, use the MiniZinc IDE, or the command line as in the following example:

minizinc wallace-hoist_cpaior20-submission.mzn --solver chuffed -v -p 1 -O1 -d dzn/1-1-1.dzn

The output of the above command (MiniZinc 2.4.3, Chuffed 0.10.4) is:

multiplier 1
hoists [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
removal times [0, 195, 324, 485, 76, 131, 272, 354, 450, 378, 43, 168, 220]
sorted removal times [0, 43, 76, 131, 168, 195, 220, 272, 324, 354, 378, 450, 485]
period 521
jobs 3
----------
==========
