function q5_fun = q5
  q5_fun.q5c=@q5c;
return

# rest = -1, medicine = 1

function [w] = q5c()
    A = [-160, -20, -1; -180, -25, -1; -160, -40, -1; 180, 35, 1]
    u = [0;0;0]
    v=[1;1;1;1]
    w=linprog(-u,-A,-v)
return

function [w] = q5e()
    A = [160, 20, 1; -170, -20, -1; 180, 20, 1]
    u = [0;0;0]
    v=[1;1;1]
    w=linprog(-u,-A,-v)
return