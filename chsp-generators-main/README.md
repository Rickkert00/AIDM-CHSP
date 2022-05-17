The generators are partially based on: M. Wallace and N. Yorke-Smith (2020) A new constraint programming model and solving for the cyclic hoist scheduling problem. Constraints 25, 319–337.

contact: n.yorke-smith@tudelft.nl
released under CC BY-NC-SA license (https://creativecommons.org/licenses/by-nc-sa/4.0/)

# chsp-generators
-----------------------
1. gen_dzn_linear.py
-----------------------
It generates dzn files for instances having Loading station = Unloading station and a linear topology.
The distance between L/U and each tank increases for tanks with higher index
L/U T1 T2 T3
This is the generator suggested by previous authors.
The files are located at "../instances/linear" folder.


-----------------------
2. gen_dzn_linear_correct_f.py
-----------------------
It modifies the above dzn files to correct the last element of f array (increases the loaded time required from the last tank to the U/L station).
The files are located at "../instances/linear_correct_f" folder.

----------------------
3. gen_dzn_ring.py
----------------------
It generates dzn files for instances having Loading station = Unloading station and a ring topology.
    T1
L/U    T2
    T3
The files are located at "../instances/ring" folder.

------------------------
4. gen_dzn_reverse.py
------------------------
It generates dzn files for instances having Loading station = Unloading station and a linear topology.
The distance between L/U and each tank decreases for tanks with higher index
L/U T3 T2 T1
The files are located at "../instances/reverse" folder.