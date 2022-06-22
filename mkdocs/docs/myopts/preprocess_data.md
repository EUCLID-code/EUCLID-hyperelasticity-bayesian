#### `extractSystemOfEquations(fem_dir,loadsteps):`

Processes nodal and element data at various load steps to assemble the linear equation [A1;lambda_r*A2]*theta = [b1;lambda_r*b2]

_Input Arguments_

-`fem_dir` - File path location for data from FEM simulations
   
-`loadsteps` - The deformation steps from the FEM simulations used to discover the material properties
   
---
   