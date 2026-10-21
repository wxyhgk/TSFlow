# Runtime Assets

Static helper programs bundled with the backend live here. They are invoked by
format adapters and are not Python application modules.

Currently `sdf_distance_geometry.mjs` generates an initial 3D conformation from
a 2D SDF input. Keep model weights and generated calculation data out of this
directory.
