
default_kurbeltrieb = {
"geometrie": {
    "kurbelradius":              20.000,         # mm
    "pleuellaenge":              71.400,         # mm
    "zylinderdurchmesser":       54.000,         # mm
    "lagerabstand z-pos":        23.500,         # mm
    "lagerabstand z-neg":        22.500,         # mm
    "hubzapfenlager-pleuel cog": 31.159,         # mm
    "excentrizitaet":             3.692,         # mm
    },
"masse": {
    "kurbelwelle":              598.381E-6,      # t
    "pleuel":                    85.275E-6,      # t
    "kolben":                   139.414E-6,      # t
    "kurbelwelle inertia":    95174.381E-6,      # t * mm ^ 2
    "pleuel inertia":             0.000E-6,      # t * mm ^ 2
    "kolben inertia":             0.000E-6,      # t * mm ^ 2
    },
"motor": {
    "bauart":                     "2-Takt",
    "drehrichtung":                     -1,      # negative Z
    "ordnungen":                    [1, 2],
    },
"unwucht": {
    "masse":                    440.E-6,         # t * mm
    "winkel":                    -3.141592 / 3., # rad - angle from hubzapfenlager
    },
}

