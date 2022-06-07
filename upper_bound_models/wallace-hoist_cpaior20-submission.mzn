%
% Hoist scheduling model - M hoists 1 track
%
%
% Note: the CPAIOR'20 reviewers observed that constraint:
%
%   constraint forall(i in 1..N)((r[i])*B[i] <= (r[i-1]+f(i-1))) ; %-20)) ;   % Hoists at same tank - add delays while hoists go up and down
%
% should be replaced by constraint:
%
%   constraint forall(i in 1..N)(r[i] <= r[i-1]+f(i-1) + p_ub*(Capacity-B[i])) ;
%
% to handle the case of non-boolean B's;
% this is used in the Constraints journal article
%
%
% accompanies CPAIOR 2020 abstract "A New Constraint Programming Model and Solving for the Cyclic Hoist Scheduling Problem", and Constraints journal article of the same title
%
% Copyright (c) 2020 M. Wallace and N. Yorke-Smith
% contact: n.yorke-smith@tudelft.nl
% released under CC BY-NC-SA license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
%

include "globals.mzn";
% include "PU.dzn" ;
include "B01.dzn" ;


%%% parameters %%%

int: Multiplier ; % Benchmark problem size
int: Hoists ;   %Try multiple hoists
int: Capacity ; % Try multi-job capacity tanks

int: J;  % number of simult. jobs
int: Ninner;  %  number of tanks
int: Tinner = Ninner+1;  % number of treatments
int: N = Multiplier*Ninner ;

array [1..Ninner] of int: tmin ;
array [1..Ninner] of int: tmax ;
array [1..Tinner,0..Ninner] of int: e ;
array [0..Ninner] of int: f ;

int: INF = 9999 ;  % infinity


%%% variables %%%

array[0..N] of var 0..p_ub: r;  % removal time from tank i
array[0..N] of var 0..p_ub: r_ws;
constraint forall(i in 0..N)( r[i] <= p );

var p_lb..p_ub: p;              % cycle period
array [0..N] of var 1..Hoists: hoist ;
constraint hoist[0]=1 ;


%%% constraints %%%

array [1..N] of var 0..Capacity:B ;
constraint sum(B) <=J ;

constraint forall(i in 1..N) (r[i] <= r_ws[i]);
constraint forall(i in 1..N)(r[i]+p*B[i] >= r[i-1]+f(i-1)+tmin(i)) ;
constraint forall(i in 1..N)( r[i]+p*B[i] <= r[i-1]+f(i-1)+tmax(i)) ;

constraint p >= r[N]+f(N) ;
constraint r[0] = 0; 

constraint forall(i in 1..N,j in 0..i-1)(hoist[i] > hoist[j] \/
              ( (r[i]+f(i)+e(i+1,j)<=r[j] \/ r[j]+f(j)+e(j+1,i)<=r[i]) /\
                 r[i]+f(i)+e(i+1,j)<=r[j] + p /\
                 r[j]+f(j)+e(j+1,i)<=r[i] + p
              )
           );
                                                                                                                                 
constraint forall(i in 1..N)((r[i])*B[i] <= (r[i-1]+f(i-1))) ; %-20)) ;   % Hoists at same tank - add delays while hoists go up and down


%%% bounds %%%
function int: next(int:k) =  (if k=N then 0 else tmin(k+1) endif) ;
int: p_lb =  sum(k in 1..N)(f(k)+
                            min([next(k)]++[e(k+1,j)|j in 0..N where j != k+1])) div Hoists;
int: p_ub = sum([reverse(sort([tmin(k)+f(k)|k in 1..N]))[k]|k in 1..N div Hoists]) ;


%%% solve %%%
solve 
   :: warm_start_array([
     warm_start(r_ws[0..24], [0, 4459, 5448, 6697, 7549, 8936, 10407, 11427, 13008, 13974, 15301, 17907, 20453, 23112, 23996, 25446, 26176, 26963, 28756, 29542, 30930, 31776, 32846, 35452, 40000])
   ])
minimize p ;

       
%%% output %%%
output ["multiplier ", show(Multiplier), "\n"] ++ 
       ["hoists ", show(hoist), "\n"] ++
       ["removal times ", show(r), "\n"] ++
       ["sorted removal times ", show(sort(r)),"\n"] ++
       ["period ", show(p)] ++
       ["\njobs ", show(sum(B))] ;


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