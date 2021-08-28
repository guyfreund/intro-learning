function q3_fun = q3
  q3_fun.q3e=@q3e;
return

function [] = q3e()
  X = 0:0.1:1;
  Y = 2.*X.*(1-X);
  X2 = 0:0.1:0.5
  Y2 = 0:0.1:0.5
  X3 = 0.5:0.1:1
  Y3 = 1-X3
  X2 = [X2 X3];
  Y2 = [Y2 Y3];
  figure;
  plot(X,Y,X2,Y2);
  title('Q3.e');
  xlabel('alpha');
  ylabel('f(alpha)');
return