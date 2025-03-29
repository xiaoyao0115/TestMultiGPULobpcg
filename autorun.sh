cd /workspace/work_data/MatrixData/structural/mtx
# for file in Cube_Coup_dt6 Transport Flan_1565 Hook_1498 Long_Coup_dt0 Geo_1438 Emilia_923 CoupCons3D msdoor halfb pwtk shipsec5 consph apache1 oilpan pkustk03 cant srb1 pct20stif bcsstk39 pdb1HYS  ; do
for file in pkustk03 cant srb1 pct20stif bcsstk39 pdb1HYS  ; do

mpiexec -n 1 /workspace/work_data/TestMultiGPULobpcg/build/TestLobpcg -fileA /workspace/work_data/MatrixData/structural/petsc/${file}.petsc
mpiexec -n 2 /workspace/work_data/TestMultiGPULobpcg/build/TestLobpcg -fileA /workspace/work_data/MatrixData/structural/petsc/${file}.petsc
mpiexec -n 4 /workspace/work_data/TestMultiGPULobpcg/build/TestLobpcg -fileA /workspace/work_data/MatrixData/structural/petsc/${file}.petsc

done