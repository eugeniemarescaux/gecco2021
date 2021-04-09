# gecco2021
The script `script-popt-computation.py` were used in combination to the following bash commands. The functions `expes` and `expes11` respectively corresponds to the cases r=[1,1] and r=[11,11]. They take two arguments, pmin and pmax, which are the bounds on the p for which the p-optimal distribution is computed.
```bash
function pretty_exp() {
	nohup nice $1 > output_$2.txt 2> errors_$2.txt &
}

function expes() {
   for fun in convex-biL convex-doublesphere convex-zdt1 concave-biL concave-dtlz2 concave-zdt2
   do
       pretty_exp "python -m script-popt-computation --opt=SLSQP --pmin=$1 --pmax=$2 --fun=$fun --nb_restarts=3" $fun
   done
}
function expes11() {
   for fun in convex-biL convex-doublesphere convex-zdt1 concave-biL concave-dtlz2 concave-zdt2
   do
       pretty_exp "python -m script-popt-computation --opt=CMA --pmin=$1 --pmax=$2 --fun=$fun --r1=11 --r2=11 --nb_restarts=3" $fun-11-11
   done
}
```
