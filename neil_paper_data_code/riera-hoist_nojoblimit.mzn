%
% Hoist scheduling model - M hoists 1 track
%
% implementation of Riera & Yorke-Smith (2002) hoist model
%
% accompanies CPAIOR 2020 abstract "A New Constraint Programming Model and Solving for the Cyclic Hoist Scheduling Problem", and Constraints journal article of the same title
%
% Copyright (c) 2020 M. Wallace and N. Yorke-Smith
% contact: n.yorke-smith@tudelft.nl
% released under CC BY-NC-SA license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
%

include "globals.mzn";
include "PU.dzn" ;
%include "BO1.dzn" ;


%%% parameters %%%

int: Multiplier ;
int: Hoists ;
int: Ninner ;
int: Tinner = Ninner+1 ;

int: N = Multiplier*Ninner ;  % number of tanks
% Fixed one track
int: R = 1;  % number of tracks
int: H = Hoists;  % number of hoists
int: J;  % number of simult. jobs
int: T = N + 1;  % number of treaments
int: Capacity ;

int: INF = 9999 ;  % infinity
int: BIGM = 99999 ;  % big-M constant

array[0..T] of 0..T: s = array1d(0..T, [i | i in 0..T]) ;  % tank for i^th treatment
array[1..N] of int: c = [Capacity| i in 1..N];  % capacity of j^th tank
array[1..Ninner] of int: tmin;  % min treatment time in tank j
array[1..Ninner] of int: tmax;  % max treatment time in tank j

array[1..Ninner+1,0..Ninner] of int: e;  % empty travel times from tank j to k
array[0..Ninner] of int: f;  % full travel times from tank j to k


%%% variables %%%

array[0..N] of var 0..r_upper_bound: r;  % removal time at completion of i^th treatment
array[0..N] of var 1..H: b;  % number of the hoist that performs the i^th treatment
var 0..mymaxvalue: p;  % cycle period

array[0..N,0..N] of var 0..1: C;  % C_ij = whether same hoist handles tank i and tank j
array[1..T] of var int: t;  % actual time of i^th treatment -- not a decision variable because can be inferred

       
%%% constraints %%%

int: mymaxvalue = f(0) + sum(k in 1..T-1)( tmin( s[k]) + f(k) ) ;

constraint r[0] = 0;

constraint forall (i in 1..N, j in i..N) ( C[i,j] = 1 );  % unused C variables

constraint forall (i in 1..N) ( t[i] = r[i] - (r[i-1] + f(i-1)) );  % (1)

constraint r[1] <= f(0) + p * c[ s[1]];  % (2a)
constraint forall (i in 2..N) ( r[i] <= p * c[ s[i]] + r[i-1] + f(i-1) );  % (2b)

constraint forall (i in 1..N) ( tmin(s[i]) <= t[i] );  % (lhs4)
constraint forall (i in 1..N) ( tmax(s[i]) >= t[i] );  % (rhs4)
constraint forall (i in 1..N, j in 0..i-1) (
        (b[ s[i]] < b[ s[j]]) <- C[i,j] = 1   
    );  % (9)

predicate no_overlap_MM(var int: Ri, var int: Fi, var int: Eip1j, 
                        var int: Rj, int: k, var int: p,
                        var int: Fj, var int: Ejp1i,
                        var 0..1: myC,
                        par int: i, par int: j) = 
          ((Ri + Fi + Eip1j         <= Rj + k * p + myC   * BIGM) \/ 
          (Rj + Fj + Ejp1i + k * p <= Ri +             myC * BIGM) );  

constraint forall (i in 1..N, j in 0..i-1, k in 1..(J-1)) (
        no_overlap_MM(r[i], f(i), e( s[i+1],s[j]), 
                      r[j], k, p, 
                      f(j), e( s[j+1],s[i]),
                         C[i,j], i, j)
    );


%%% redundant constraints %%%

% break symmetries about hoist-to-tank assignment
%constraint b[0] = H;


%%% bounds %%%

constraint forall (i in 1..N)( tmin[1] <= r[i] );  % (lhs16)
par int: r_upper_bound = sum(k in 1..T-1)( tmax( s[k]) ) + 
                         sum(k in 0..T-2)( f(k) );


%%% solve %%%
solve 
    minimize p;


%%% output %%%
output [
  "removal times ", show(r[0..T-1]), 
  "\ntreatments by hoists ", show(b), 
  "\nHoists ", show(H),
  "\nperiod ", show(p)];
  
  
%%% helper predicates %%%

% This is the core function used by tmin, tmax, f and even e
% It is "x mod Ninner", adjusting "mod" to handle 1..Ninner instead of 0..Ninner-1
function int:multtank(int:x) =
    ((x-1) mod Ninner)+1 ;
 
% The standard mapping applies to tmin and tmax      
function int: tmin(int: i) =
    let { int: ic = multtank(i)
    }      in
    tmin[ic];
function int: tmax(int: i) =
    let { int: ic = multtank(i)
      } in    
     tmax[ic];
        
% e is different because it is e(1..Tinner,0..Ninner)  
% This version handles the extra time for moves from the m1th copy of tank i to the m2th copy of tank j
function int: e(int: i, int:j)  =
    let { 
            int: ic = multempty(i),
            int: jc =  mult2empty(j),
            int: cyclei = cycleempty(i),
            int: cyclej = cycle2empty(j)           
         }
    in         5*abs(cyclei-cyclej) + e[ic,jc]; 
% This is the standard mapping extended to handle  e(y,N+1)
function int: multempty(int:x) =
    if    x=N+1 then Tinner
    else   multtank(x) endif ;
% This is the standard mapping extended to handle e(0,y)
function int: mult2empty(int:x) =
   if x=0 then 0 
    else multtank(x) endif ;  
% This calculates which copy, of the original Ninner tanks, x belongs to
function int: cycleempty(int:x) =
    if x=N+1 then Multiplier-1
    else (x-1) div Ninner endif;    
% This calculates which copy x belongs to  
function int: cycle2empty(int:x) =
   if x=0 then 0
   else (x-1) div Ninner endif;

% f is almost the same, except that it includes tank 0    
% This is the standard mapping extended to handle x=0
function int: f(int: i) =
    let { int: ic = mult2empty(i) } 
    in
    f[ic] ;