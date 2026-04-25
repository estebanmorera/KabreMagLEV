///////////////////////////////////////////////////////////
//   HMB / PMB MASTER GEOMETRY - AUTO-CENTRADO EN Z
//   Rotor = imanes internos (barridos dr / dz)
//   Compatible con Gmsh -> ElmerGrid -> ElmerSolver
///////////////////////////////////////////////////////////

SetFactory("OpenCASCADE");

// =======================================================
// FLAGS Y BARRIDOS
// =======================================================
use_coil = 1;        // 1 = HMB (con bobina), 0 = PMB-only

dx = 0.0;            // [m] desplazamiento en X del rotor
dy = 0.0;            // [m] desplazamiento en Y del rotor
dz = 0.0;            // [m] desplazamiento en Z del rotor


// =======================================================
// PARAMETROS PMB (Individuo 251)
// =======================================================
r1 = 6.5e-3;         // radio interno imanes interiores
r2 = 11.5e-3;         // radio externo imanes interiores
r3 = 12.7e-3;         // radio interno imanes exteriores
r4 = 25.4e-3;         // radio externo imanes exteriores

h0 = 3.2e-3;       // altura imanes interiores
h1 = 3.4e-3;       // altura imanes exteriores

gap_z = 1.0e-3;      // separacion axial entre imanes interiores


// =======================================================
// PARAMETROS AMB (Bobina axial)
// =======================================================
rb1 = 12.7e-3;      // radio interno bobina
rb2 = 18.7e-3;      // radio externo bobina
hb  = 12.7e-3;      // altura bobina

gap_pc = 4.0e-3;     // separacion axial PMB-bobina


// =======================================================
// AUTO-CENTRADO AXIAL DEL STACK
// =======================================================
H_pmb   = 2*h0 + gap_z;
H_total = H_pmb + gap_pc + hb;

z_pmb_bottom = -H_total/2;   // TODO el sistema centrado en Z=0


// =======================================================
// DOMINIO DE AIRE
// =======================================================
margin_air = 10e-3;
air_H = H_total + 2*margin_air;
air_R = 40.0e-3;


// =======================================================
// 1) IMANES INTERNOS (ROTOR)
// =======================================================

// --- Inferior ---
z_mi1 = z_pmb_bottom;

Cylinder(5) = {dx, dy, z_mi1+dz,  0,0,h0,  r2, 2*Pi};
Cylinder(6) = {dx, dy, z_mi1+dz,  0,0,h0,  r1, 2*Pi};
MagIntInf[] = BooleanDifference{ Volume{5}; Delete; }{ Volume{6}; Delete; };
InnerMag1 = MagIntInf[0];

// --- Superior ---
z_mi2 = z_mi1 + h0 + gap_z;

Cylinder(7) = {dx, dy, z_mi2+dz,  0,0,h0,  r2, 2*Pi};
Cylinder(8) = {dx, dy, z_mi2+dz,  0,0,h0,  r1, 2*Pi};
MagIntSup[] = BooleanDifference{ Volume{7}; Delete; }{ Volume{8}; Delete; };
InnerMag2 = MagIntSup[0];


// =======================================================
// 2) IMANES EXTERIORES (ESTATOR)
// =======================================================
z_me1 = z_mi1 + (h0 - h1)/2;

Cylinder(1) = {0,0,z_me1,  0,0,h1,  r4, 2*Pi};
Cylinder(2) = {0,0,z_me1,  0,0,h1,  r3, 2*Pi};
MagExtInf[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };
OuterMag1 = MagExtInf[0];

z_me2 = z_me1 + h0 + gap_z;

Cylinder(3) = {0,0,z_me2,  0,0,h1,  r4, 2*Pi};
Cylinder(4) = {0,0,z_me2,  0,0,h1,  r3, 2*Pi};
MagExtSup[] = BooleanDifference{ Volume{3}; Delete; }{ Volume{4}; Delete; };
OuterMag2 = MagExtSup[0];


// =======================================================
// 3) BOBINA AXIAL (AMB)
// =======================================================
If (use_coil)

  z_coil = z_pmb_bottom + H_pmb + gap_pc;

  Cylinder(20) = {0,0,z_coil,  0,0,hb,  rb2, 2*Pi};
  Cylinder(21) = {0,0,z_coil,  0,0,hb,  rb1, 2*Pi};
  CoilVol[] = BooleanDifference{ Volume{20}; Delete; }{ Volume{21}; Delete; };
  Coil = CoilVol[0];

Else
  Coil = -1;
EndIf


// =======================================================
// 4) AIRE (RESTANDO SOLIDOS)
// =======================================================
z_air = -air_H/2;
Cylinder(30) = {0,0,z_air,  0,0,air_H,  air_R, 2*Pi};

If (use_coil)
  AirClean[] = BooleanDifference{ Volume{30}; Delete; }{
    Volume{OuterMag1, OuterMag2, InnerMag1, InnerMag2, Coil};
  };
Else
  AirClean[] = BooleanDifference{ Volume{30}; Delete; }{
    Volume{OuterMag1, OuterMag2, InnerMag1, InnerMag2};
  };
EndIf

Air = AirClean[0];

Coherence;


// =======================================================
// DEBUG: IDs REALES (para scripting)
// =======================================================
Printf("DEBUG OuterMag1 = %g", OuterMag1);
Printf("DEBUG OuterMag2 = %g", OuterMag2);
Printf("DEBUG InnerMag1 = %g", InnerMag1);
Printf("DEBUG InnerMag2 = %g", InnerMag2);
Printf("DEBUG Coil      = %g", Coil);
Printf("DEBUG Air       = %g", Air);

// =======================================================
// OPCIONES 3D / OPTIMIZACION
// =======================================================
Mesh.Algorithm3D = 10;      // HXT
Mesh.Smoothing   = 8;
Mesh.Optimize    = 1;
Mesh.OptimizeNetgen = 1;
Mesh.OptimizeThreshold = 0.45;

// =======================================================
// TAMANOS
// =======================================================
lc_far   = 3.0e-3;
lc_mid2  = 1.8e-3;
lc_mid1  = 1.0e-3;
lc_near  = 0.6e-3;
lc_gap   = 0.35e-3;

Mesh.CharacteristicLengthMin = lc_gap;
Mesh.CharacteristicLengthMax = lc_far;

// =======================================================
// REFINAMIENTO QUE SIGUE AL ROTOR
// =======================================================
rotorR    = r2;
rotorZmin = z_mi1 + dz;
rotorZmax = z_mi2 + dz + h0;

// caja muy fina
Field[10] = Box;
Field[10].VIn  = lc_gap;
Field[10].VOut = lc_far;
Field[10].XMin = dx - (rotorR + 2.5e-3);
Field[10].XMax = dx + (rotorR + 2.5e-3);
Field[10].YMin = dy - (rotorR + 2.5e-3);
Field[10].YMax = dy + (rotorR + 2.5e-3);
Field[10].ZMin = rotorZmin - 1.5e-3;
Field[10].ZMax = rotorZmax + 1.5e-3;

// caja cercana
Field[11] = Box;
Field[11].VIn  = lc_near;
Field[11].VOut = lc_far;
Field[11].XMin = dx - (rotorR + 5.0e-3);
Field[11].XMax = dx + (rotorR + 5.0e-3);
Field[11].YMin = dy - (rotorR + 5.0e-3);
Field[11].YMax = dy + (rotorR + 5.0e-3);
Field[11].ZMin = rotorZmin - 3.0e-3;
Field[11].ZMax = rotorZmax + 3.0e-3;

// caja intermedia
Field[12] = Box;
Field[12].VIn  = lc_mid1;
Field[12].VOut = lc_far;
Field[12].XMin = -20e-3;
Field[12].XMax =  20e-3;
Field[12].YMin = -20e-3;
Field[12].YMax =  20e-3;
Field[12].ZMin = -18e-3;
Field[12].ZMax =   2e-3;

// caja de transicion lejana
Field[13] = Box;
Field[13].VIn  = lc_mid2;
Field[13].VOut = lc_far;
Field[13].XMin = -26e-3;
Field[13].XMax =  26e-3;
Field[13].YMin = -26e-3;
Field[13].YMax =  26e-3;
Field[13].ZMin = -22e-3;
Field[13].ZMax =   6e-3;

Field[14] = Min;
Field[14].FieldsList = {10,11,12,13};
Background Field = 14;

// ===================== OPCIONES DE GUARDADO =====================
Mesh.MshFileVersion = 2.2;
Mesh.Binary         = 0;
Mesh.SaveParametric = 0;
Mesh.SaveAll        = 1;

Mesh 3;
