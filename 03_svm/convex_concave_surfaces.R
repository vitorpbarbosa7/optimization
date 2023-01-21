

saddle = function(x,y){
  -x*x + y*y
}

convex = function(x,y){
  x*x + y*y
}

x = y = seq(-1, +1, length = 100)
z_saddle = outer(x,y,saddle)
z_convex = outer(x,y,convex)

persp(x,y,z_saddle)
persp(x,y,z_convex)