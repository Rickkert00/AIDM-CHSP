%
% Hoist scheduling model - M hoists 1 track
%
% implementation of Che et al (2014) hoist model
%
% accompanies CPAIOR 2020 abstract "A New Constraint Programming Model and Solving for the Cyclic Hoist Scheduling Problem", and Constraints journal article of the same title
%
% Copyright (c) 2020 M. Wallace and N. Yorke-Smith
% contact: n.yorke-smith@tudelft.nl
% released under CC BY-NC-SA license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
%

include "PU.dzn" ;
%include "BO1.dzn" ;

%% warm start parameters %%%

int: heur_len;
array[0..heur_len] of int: r_heur;

int: Multiplier ;
int: Ninner ;
int: n = Ninner*Multiplier;
int: Tinner = Ninner+1 ;
int: Capacity ;  % defined as 1 in P&U
int: J;  % number of jobs (although not used in this model)

set of int: N = 1..n ;
set of int: N0 = 0..n ;
int: K = Hoists ;
int: Hoists ;
set of int: KCurly = 1..K ;

array [N0] of int: d = array1d(N0,[f(i) | i in N0]) ;
array [0..Ninner] of int: f ;

array [1..Tinner, 0..Ninner] of int: e ;
array [1..n+1,N0] of int: ee =
      array2d(1..n+1,N0, [e(i,j)|i in 1..n+1,j in N0]) ;
array [N] of int:L = [tmin(i)|i in N] ;
array [1..Ninner] of int: tmin ;
array [N] of int: U = [tmax(i)|i in N] ;
array [1..Ninner] of int: tmax ;
float: M = 999999.0 ;
float: delta = 0.001 ;
int: INF = 99999;

float: maxCycle = 10000.0 ;
array [N0] of var 0.0..maxCycle: t ;
array [N0,N0] of var 0..1:y ;
%array [N0] of var 0..1: LCurly ;
array [N0,KCurly] of var 0..1 : z ;
array [N] of var 0..1: s ;
array [N0] of var 0..1: w ;
var 0.0..maxCycle: C ;

float: p = 0 ;  % for P&U  8.5 ;  % Time required to unload a part from a tank
float: q = 0 ;  % for P&U 11.5 ;  % Time required to load a part into a tank


%%% constraints %%%

% 1,7,8,47
constraint forall(i in N)(sum([z[i,k]| k in KCurly])=1) ;
constraint forall(j in N)(t[j]-(d[0]+ee[1,j])*z[j,1] >= 0) ;
constraint t[0]=0 ;
constraint forall(i in N0)(t[i]+(d[i]+ee[i+1,0])*z[i,1] <= C) ;

% constraint for w
constraint forall(i in N)((t[i]< C /\ t[i]+d[i] < C) <-> w[i]=0) ;
%29-30
constraint forall(i in N)(t[i] - (t[i-1]+d[i-1]) <= U[i] + M*(s[i]+w[i-1]) ) ;
constraint forall(i in N)(t[i] - (t[i-1]+d[i-1]) >= L[i] - M*(s[i]+w[i-1]) ) ;
% %31-32
constraint forall(i in N)(C + t[i] - (t[i-1]+d[i-1]) <= U[i] + M*((1-s[i])+w[i-1]) ) ;
constraint forall(i in N)(C + t[i] - (t[i-1]+d[i-1]) >= L[i] - M*((1-s[i])+w[i-1]) ) ;
% %33-34
constraint forall(i in N)(t[i] - (t[i-1]+d[i-1]-C) <= U[i] + M*(s[i]+(1-w[i-1])) ) ;
constraint forall(i in N)(t[i] - (t[i-1]+d[i-1]-C) >= L[i] - M*(s[i]+(1-w[i-1])) ) ;
% %35-36
constraint forall(i in N)(C+t[i] - (t[i-1]+d[i-1]-C) <= U[i] + M*((2-w[i-1])-s[i]) ) ;
constraint forall(i in N)(C+t[i] - (t[i-1]+d[i-1]-C) >= L[i] -   M*((2-w[i-1])-s[i]) ) ;
% %37-39
constraint forall(i in N)((t[i]+p)-(t[i-1]+d[i-1]-q) <= M*((1-s[i])+w[i-1])) ;
constraint forall(i in N)((t[i]+p)-(t[i-1]+d[i-1]-q) <= M*((1-w[i-1])+s[i])) ;
constraint forall(i in N)((t[i]+p)-((t[i-1]+d[i-1]-q)-C) <= M*((2-w[i-1])-s[i])) ;

% % 14, 15
constraint forall(i,j in N where i != j)(t[i]-t[j] <= M*y[i,j]) ;
constraint forall(i,j in N where i != j)(y[i,j]+y[j,i]=1) ;

%16-21
constraint forall(i,j in N,k in KCurly where j<i)(t[i]+d[i]+ee[i+1,j]-t[j] <= M*(3 -(y[j,i]+z[i,k]+sum([z[j,h]|h in k..K])))) ;
constraint forall(i,j in N,k in KCurly where j<i)(t[j]+d[j]+ee[j+1,i]-t[i] <= M*(3 -(y[i,j]+z[i,k]+sum([z[j,h]|h in k..K])))) ;
% %18
constraint forall(i,j in N,k in KCurly where i<j)(t[j]+d[j]+ee[j+1,i]-t[i] <= M*(3 -(y[i,j]+z[i,k]+sum([z[j,h]|h in 1..k])))) ;
constraint forall(i,j in N,k in KCurly where i<j)(t[i]+d[i]+ee[i+1,j]-t[j] <= M*(3 -(y[j,i]+z[i,k]+sum([z[j,h]|h in 1..k])))) ;

constraint forall(i,j in N,k in KCurly where j<i)(t[j]+d[j]+ee[j+1,i]-(C+t[i]) <= M*(2 -(z[i,k]+sum([z[j,h]|h in k..K])))) ;
constraint forall(i,j in N,k in KCurly where j<i)(t[i]+d[i]+ee[i+1,j]-(C+t[j]) <= M*(2 -(z[i,k]+sum([z[j,h]|h in k..K])))) ;

% 41-46
constraint forall(i in N)(t[i]+d[i] <= C+M*w[i]) ;
constraint forall(i in N)(t[i]+d[i] >= C- M*(1-w[i])) ;
constraint forall(i in N0)(w[i]+z[i,1] <= 1) ;
% w[i] in 0..1

% 24, 26, 27, 44
% z[i] in 0..1
% s[i] in 0..1
% y[i,j] in 0..1
% w[i] in 0..1


%%% solve %%%
solve 
  :: warm_start_array([
  warm_start(t[0..heur_len], r_heur[0..heur_len]),
  ])
minimize C ;


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
    if    x=n+1 then Tinner
    else   multtank(x) endif ;
% This is the standard mapping extended to handle e(0,y)
function int: mult2empty(int:x) =
   if x=0 then 0 
    else multtank(x) endif ;  
% This calculates which copy, of the original Ninner tanks, x belongs to
function int: cycleempty(int:x) =
    if x=n+1 then Multiplier-1
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