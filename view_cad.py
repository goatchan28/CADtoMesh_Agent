import gmsh
gmsh.initialize()
gmsh.model.occ.importShapes("models/acorn_nut.step")
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Geometry.SurfaceLabels", 1)
gmsh.option.setNumber("Geometry.CurveLabels", 1)
gmsh.fltk.run()          # blocks until you close the window
gmsh.finalize()
