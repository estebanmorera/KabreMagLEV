SetFactory("OpenCASCADE");

use_coil = 1;
dx = 0;
dy = 0;
dz = 0;
r1 = 0.0065;
r2 = 0.0115;
r3 = 0.0127;
r4 = 0.0254;
r_core = 0.0052;
h0 = 0.0032;
h1 = 0.0034;
gap_z = 0.001;
rb1 = 0.0127;
rb2 = 0.0187;
hb  = 0.0127;
gap_pc = 0.004;

H_pmb = 2*h0 + gap_z;
H_total = H_pmb + gap_pc + hb;
z_pmb_bottom = -H_total/2;
air_R = 0.05;
air_H = 0.06;
z_air = -air_H/2;

z_mi1 = z_pmb_bottom;
z_mi2 = z_mi1 + h0 + gap_z;
Cylinder(5) = {dx, dy, z_mi1+dz,  0,0,h0,  r_core, 2*Pi};
InnerMag1 = 5;

Cylinder(7) = {dx, dy, z_mi2+dz,  0,0,h0,  r_core, 2*Pi};
InnerMag2 = 7;

z_me1 = z_mi1 + (h0 - h1)/2;
Cylinder(1) = {0,0,z_me1, 0,0,h1, r4, 2*Pi};
Cylinder(2) = {0,0,z_me1, 0,0,h1, r3, 2*Pi};
MagExtInf[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };
OuterMag1 = MagExtInf[0];
z_me2 = z_me1 + h0 + gap_z;
Cylinder(3) = {0,0,z_me2, 0,0,h1, r4, 2*Pi};
Cylinder(4) = {0,0,z_me2, 0,0,h1, r3, 2*Pi};
MagExtSup[] = BooleanDifference{ Volume{3}; Delete; }{ Volume{4}; Delete; };
OuterMag2 = MagExtSup[0];

z_coil = z_pmb_bottom + H_pmb + gap_pc;
Cylinder(20) = {0,0,z_coil, 0,0,hb, rb2, 2*Pi};
Cylinder(21) = {0,0,z_coil, 0,0,hb, rb1, 2*Pi};
CoilVol[] = BooleanDifference{ Volume{20}; Delete; }{ Volume{21}; Delete; };
Coil = CoilVol[0];

Cylinder(30) = {0,0,z_air, 0,0,air_H, air_R, 2*Pi};
AirClean[] = BooleanDifference{ Volume{30}; Delete; }{ Volume{OuterMag1, OuterMag2, InnerMag1, InnerMag2, Coil}; };
Air = AirClean[0];
Coherence;

Physical Volume("OuterMag1", 1) = {OuterMag1};
Physical Volume("OuterMag2", 3) = {OuterMag2};
Physical Volume("InnerMag1", 5) = {InnerMag1};
Physical Volume("InnerMag2", 7) = {InnerMag2};
Physical Volume("Coil", 20) = {Coil};
Physical Volume("Air", 30) = {Air};

Mesh.Algorithm3D = 10;
Mesh.Smoothing = 8;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.OptimizeThreshold = 0.45;
lc_inner = 0.00028;
lc_coil = 0.0004;
lc_near = 0.00075;
lc_mid = 0.0015;
lc_far = 0.003;
Mesh.CharacteristicLengthMin = lc_inner;
Mesh.CharacteristicLengthMax = lc_far;

Field[10] = Box;
Field[10].VIn = lc_inner; Field[10].VOut = lc_far;
Field[10].XMin = -0.0077; Field[10].XMax = 0.0077;
Field[10].YMin = -0.0077; Field[10].YMax = 0.0077;
Field[10].ZMin = -0.01355; Field[10].ZMax = -0.00315;

Field[11] = Box;
Field[11].VIn = lc_coil; Field[11].VOut = lc_far;
Field[11].XMin = -0.0207; Field[11].XMax = 0.0207;
Field[11].YMin = -0.0207; Field[11].YMax = 0.0207;
Field[11].ZMin = -0.00265; Field[11].ZMax = 0.01405;

Field[12] = Box;
Field[12].VIn = lc_near; Field[12].VOut = lc_far;
Field[12].XMin = -0.028; Field[12].XMax = 0.028;
Field[12].YMin = -0.028; Field[12].YMax = 0.028;
Field[12].ZMin = -0.026; Field[12].ZMax = 0.02;

Field[13] = Box;
Field[13].VIn = lc_mid; Field[13].VOut = lc_far;
Field[13].XMin = -0.04; Field[13].XMax = 0.04;
Field[13].YMin = -0.04; Field[13].YMax = 0.04;
Field[13].ZMin = -0.034; Field[13].ZMax = 0.03;

Field[14] = Min;
Field[14].FieldsList = {10,11,12,13};
Background Field = 14;
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
Mesh.SaveParametric = 0;
Mesh.SaveAll = 1;
Mesh 3;
