#include "D0ToKspipi2018.h"
#include "EvtGenBase/Resonance.hh"
//#include "Resonance.cxx"
#include <iostream>
#include <string>
#include <utility>
#include <stdlib.h>
#include <complex>
#include "TComplex.h"
#include <vector>
#include <math.h>
#include "TMath.h"
#include <sys/stat.h>
#include <sys/types.h>

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/TwoVector.h"
using CLHEP::HepVector;
using CLHEP::Hep3Vector;
using CLHEP::Hep2Vector;
using CLHEP::HepLorentzVector;
using namespace std;
const double mD0 = 1.8648399;
const double mKs = 0.49761401;
const double mPi = 0.13957017;

D0ToKspipi2018::~D0ToKspipi2018(){}

void D0ToKspipi2018::init(){
std::cout << "D0ToKSpipi2018 ==> Initialization !" << std::endl;
  _nd = 3;
  std::string m_Dalitzmodel = "KMatrix";
  std::string parameter_filename, ratio_filename, outpath;
	
_parameter_number[0] = 0; _parameter_value[0] = 0.000000000000000; _parameter_error[0] = 0.000000000000000;
_parameter_number[1] = 1; _parameter_value[1] = 0.000000000000000; _parameter_error[1] = -999.000000000000000;
_parameter_number[2] = 2; _parameter_value[2] = 0.000000000000000; _parameter_error[2] = -999.000000000000000;
_parameter_number[3] = 3; _parameter_value[3] = 0.038791318344843; _parameter_error[3] = 0.000485268275621;
_parameter_number[4] = 4; _parameter_value[4] = 2.107370203581449; _parameter_error[4] = 0.012003065245212;
_parameter_number[5] = 5; _parameter_value[5] = 1.720436206046246; _parameter_error[5] = 0.006338333542338;
_parameter_number[6] = 6; _parameter_value[6] = 2.388358772635172; _parameter_error[6] = 0.004227745091072;
_parameter_number[7] = 7; _parameter_value[7] = 2.362144656831594; _parameter_error[7] = 0.063542164457913;
_parameter_number[8] = 8; _parameter_value[8] = 1.735144928431722; _parameter_error[8] = 0.029258250508315;
_parameter_number[9] = 9; _parameter_value[9] = 1.272677892930591; _parameter_error[9] = 0.016205040955539;
_parameter_number[10] = 10; _parameter_value[10] = -0.769095387411083; _parameter_error[10] = 0.013015314144088;
_parameter_number[11] = 11; _parameter_value[11] = 3.307642176743471;  _parameter_error[11] = 0.197801960348389;
_parameter_number[12] = 12; _parameter_value[12] = -2.062227483594043; _parameter_error[12] = 0.054875902770026;
_parameter_number[13] = 13; _parameter_value[13] = 0.286927014847305;  _parameter_error[13] = 0.030500538645209;
_parameter_number[14] = 14; _parameter_value[14] = 1.734618647446857;  _parameter_error[14] = 0.096350455851831;
_parameter_number[15] = 15; _parameter_value[15] = 0.164179194147053;  _parameter_error[15] = 0.002634739389370;
_parameter_number[16] = 16; _parameter_value[16] = -0.735902683667772; _parameter_error[16] = 0.014739420816336;
_parameter_number[17] = 17; _parameter_value[17] = 0.106521701166492;  _parameter_error[17] = 0.011409380977317;
_parameter_number[18] = 18; _parameter_value[18] = 2.832844932968643;  _parameter_error[18] = 0.113040833040813;
_parameter_number[19] = 19; _parameter_value[19] = 0.102573574857500;  _parameter_error[19] = 0.012844068531610;
_parameter_number[20] = 20; _parameter_value[20] = -1.563974813878017; _parameter_error[20] = 0.133599725010340;
_parameter_number[21] = 21; _parameter_value[21] = 0.000000000000000;  _parameter_error[21] = -999.000000000000000;
_parameter_number[22] = 22; _parameter_value[22] = 0.000000000000000;  _parameter_error[22] = -999.000000000000000;
_parameter_number[23] = 23; _parameter_value[23] = 0.209032635939347;  _parameter_error[23] = 0.019466119950150;
_parameter_number[24] = 24; _parameter_value[24] = 2.620898616878612;  _parameter_error[24] = 0.091987443212371;
_parameter_number[25] = 25; _parameter_value[25] = 0.000000000000000;  _parameter_error[25] = -999.000000000000000;
_parameter_number[26] = 26; _parameter_value[26] = 0.000000000000000;  _parameter_error[26] = -999.000000000000000;
_parameter_number[27] = 27; _parameter_value[27] = 1.428870354058858;  _parameter_error[27] = 0.024844377532571;
_parameter_number[28] = 28; _parameter_value[28] = -0.633296469741719; _parameter_error[28] = 0.019041614713980;
_parameter_number[29] = 29; _parameter_value[29] = 0.000000000000000;  _parameter_error[29] = -999.000000000000000;
_parameter_number[30] = 30; _parameter_value[30] = 0.000000000000000;  _parameter_error[30] = -999.000000000000000;
_parameter_number[31] = 31; _parameter_value[31] = 2.851309898774982;  _parameter_error[31] = 0.102920356215321;
_parameter_number[32] = 32; _parameter_value[32] = 1.782080113829537;  _parameter_error[32] = 0.032555394668848;
_parameter_number[33] = 33; _parameter_value[33] = 0.000000000000000;  _parameter_error[33] = -999.000000000000000;
_parameter_number[34] = 34; _parameter_value[34] = 0.000000000000000;  _parameter_error[34] = -999.000000000000000;
_parameter_number[35] = 35; _parameter_value[35] = 0.000000000000000;  _parameter_error[35] = -999.000000000000000;
_parameter_number[36] = 36; _parameter_value[36] = 0.000000000000000;  _parameter_error[36] = -999.000000000000000;
_parameter_number[37] = 37; _parameter_value[37] = 0.000000000000000;  _parameter_error[37] = 0.000000000000000;
_parameter_number[38] = 38; _parameter_value[38] = 0.000000000000000;  _parameter_error[38] = 0.000000000000000;
_parameter_number[39] = 39; _parameter_value[39] = -0.019829903319132; _parameter_error[39] = 0.000518478383391;
_parameter_number[40] = 40; _parameter_value[40] = 0.033339785741436;  _parameter_error[40] = 0.000428325270624;
_parameter_number[41] = 41; _parameter_value[41] = -1.255025021860793; _parameter_error[41] = 0.007286396600990;
_parameter_number[42] = 42; _parameter_value[42] = 1.176780750003210;  _parameter_error[42] = 0.006323579288449;
_parameter_number[43] = 43; _parameter_value[43] = -0.386469884688245; _parameter_error[43] = 0.067291824151075;
_parameter_number[44] = 44; _parameter_value[44] = 2.330315087713914;  _parameter_error[44] = 0.065466907934166;
_parameter_number[45] = 45; _parameter_value[45] = 0.914470111251261;  _parameter_error[45] = 0.015855610898565;
_parameter_number[46] = 46; _parameter_value[46] = -0.885129049790117; _parameter_error[46] = 0.016899085045940;
_parameter_number[47] = 47; _parameter_value[47] = -1.560837188791231; _parameter_error[47] = 0.167293192561194;
_parameter_number[48] = 48; _parameter_value[48] = -2.916210561577914; _parameter_error[48] = 0.209962923071889;
_parameter_number[49] = 49; _parameter_value[49] = -0.046795079734847; _parameter_error[49] = 0.026360326062032;
_parameter_number[50] = 50; _parameter_value[50] = 0.283085379985959;  _parameter_error[50] = 0.031617913106171;
_parameter_number[51] = 51; _parameter_value[51] = 0.121693743404499;  _parameter_error[51] = 0.002764499976450;
_parameter_number[52] = 52; _parameter_value[52] = -0.110206354657867; _parameter_error[52] = 0.002270536875039;
_parameter_number[53] = 53; _parameter_value[53] = -0.101484805664368; _parameter_error[53] = 0.011298915423031;
_parameter_number[54] = 54; _parameter_value[54] = 0.032368302993344;  _parameter_error[54] = 0.012145016857679;
_parameter_number[55] = 55; _parameter_value[55] = 0.000699701539252;  _parameter_error[55] = 0.013698160252165;
_parameter_number[56] = 56; _parameter_value[56] = -0.102571188336701; _parameter_error[56] = 0.012850084622477;
_parameter_number[57] = 57; _parameter_value[57] = 0.000000000000000;  _parameter_error[57] = 0.000000000000000;
_parameter_number[58] = 58; _parameter_value[58] = 0.000000000000000;  _parameter_error[58] = 0.000000000000000;
_parameter_number[59] = 59; _parameter_value[59] = -0.181330401419455; _parameter_error[59] = 0.018781975251612;
_parameter_number[60] = 60; _parameter_value[60] = 0.103990039950039;  _parameter_error[60] = 0.019897179237575;
_parameter_number[61] = 61; _parameter_value[61] = 0.000000000000000;  _parameter_error[61] = 0.000000000000000;
_parameter_number[62] = 62; _parameter_value[62] = 0.000000000000000;  _parameter_error[62] = 0.000000000000000;
_parameter_number[63] = 63; _parameter_value[63] = 1.151785277682948;  _parameter_error[63] = 0.024057878810885;
_parameter_number[64] = 64; _parameter_value[64] = -0.845612891825272; _parameter_error[64] = 0.027905855273201;
_parameter_number[65] = 65; _parameter_value[65] = 0.000000000000000;  _parameter_error[65] = 0.000000000000000;
_parameter_number[66] = 66; _parameter_value[66] = 0.000000000000000;  _parameter_error[66] = 0.000000000000000;
_parameter_number[67] = 67; _parameter_value[67] = -0.597963342540235; _parameter_error[67] = 0.095826915382634;
_parameter_number[68] = 68; _parameter_value[68] = 2.787903868470057;  _parameter_error[68] = 0.100131808152072;
_parameter_number[69] = 69; _parameter_value[69] = 0.000000000000000;  _parameter_error[69] = 0.000000000000000;
_parameter_number[70] = 70; _parameter_value[70] = 0.000000000000000;  _parameter_error[70] = 0.000000000000000;
_parameter_number[71] = 71; _parameter_value[71] = 0.000000000000000;  _parameter_error[71] = 0.000000000000000;
_parameter_number[72] = 72; _parameter_value[72] = 0.000000000000000;  _parameter_error[72] = 0.000000000000000;
_parameter_number[73] = 73; _parameter_value[73] = 0.000000000000000;  _parameter_error[73] = 0.000000000000000;
_parameter_number[74] = 74; _parameter_value[74] = 0.000000000000000;  _parameter_error[74] = 0.000000000000000;
_parameter_number[75] = 75; _parameter_value[75] = 0.000000000000000;  _parameter_error[75] = 0.000000000000000;
_parameter_number[76] = 76; _parameter_value[76] = 0.000000000000000;  _parameter_error[76] = 0.000000000000000;
_parameter_number[77] = 77; _parameter_value[77] = 0.893709298220334;  _parameter_error[77] = 0.000056544938821;
_parameter_number[78] = 78; _parameter_value[78] = 0.047193287094108;  _parameter_error[78] = 0.000115223879792;
_parameter_number[79] = 79; _parameter_value[79] = 0.771550000000000;  _parameter_error[79] = 0.000000000000000;
_parameter_number[80] = 80; _parameter_value[80] = 0.134690000000000;  _parameter_error[80] = 0.000000000000000;
_parameter_number[81] = 81; _parameter_value[81] = 0.782650000000000;  _parameter_error[81] = 0.000000000000000;
_parameter_number[82] = 82; _parameter_value[82] = 0.008490000000000;  _parameter_error[82] = 0.000000000000000;
_parameter_number[83] = 83; _parameter_value[83] = 1.440549945739415;  _parameter_error[83] = 0.001664560546758;
_parameter_number[84] = 84; _parameter_value[84] = 0.192611512914605;  _parameter_error[84] = 0.003962024465665;
_parameter_number[85] = 85; _parameter_value[85] = 1.425600000000000;  _parameter_error[85] = 0.000000000000000;
_parameter_number[86] = 86; _parameter_value[86] = 0.098500000000000;  _parameter_error[86] = 0.000000000000000;
_parameter_number[87] = 87; _parameter_value[87] = 1.717000000000000;  _parameter_error[87] = 0.000000000000000;
_parameter_number[88] = 88; _parameter_value[88] = 0.322000000000000;  _parameter_error[88] = 0.000000000000000;
_parameter_number[89] = 89; _parameter_value[89] = 1.414000000000000;  _parameter_error[89] = 0.000000000000000;
_parameter_number[90] = 90; _parameter_value[90] = 0.232000000000000;  _parameter_error[90] = 0.000000000000000;
_parameter_number[91] = 91; _parameter_value[91] = 0.000000000000000;  _parameter_error[91] = 0.000000000000000;
_parameter_number[92] = 92; _parameter_value[92] = 0.000000000000000;  _parameter_error[92] = 0.000000000000000;
_parameter_number[93] = 93; _parameter_value[93] = 1.275100000000000;  _parameter_error[93] = 0.000000000000000;
_parameter_number[94] = 94; _parameter_value[94] = 0.184200000000000;  _parameter_error[94] = 0.000000000000000;
_parameter_number[95] = 95; _parameter_value[95] = 0.000000000000000;  _parameter_error[95] = 0.000000000000000;
_parameter_number[96] = 96; _parameter_value[96] = 0.000000000000000;  _parameter_error[96] = 0.000000000000000;
_parameter_number[97] = 97; _parameter_value[97] = 1.465000000000000;  _parameter_error[97] = 0.000000000000000;
_parameter_number[98] = 98; _parameter_value[98] = 0.400000000000000;  _parameter_error[98] = 0.000000000000000;
_parameter_number[99] = 99; _parameter_value[99] = 8.521485696272016;  _parameter_error[99] = 0.459617465922715;
_parameter_number[100] = 100; _parameter_value[100] = 1.195641256232703;  _parameter_error[100] = 0.059992453463329;
_parameter_number[101] = 101; _parameter_value[101] = 12.189520667430143; _parameter_error[101] = 0.339611757682022;
_parameter_number[102] = 102; _parameter_value[102] = 0.418025704767376;  _parameter_error[102] = 0.024267084125949;
_parameter_number[103] = 103; _parameter_value[103] = 29.146151633292522; _parameter_error[103] = 1.567406119300162;
_parameter_number[104] = 104; _parameter_value[104] = -0.001838623934792; _parameter_error[104] = 0.043391402979377;
_parameter_number[105] = 105; _parameter_value[105] = 10.745735142980751; _parameter_error[105] = 0.463388225414353;
_parameter_number[106] = 106; _parameter_value[106] = -0.905701456257495; _parameter_error[106] = 0.040291736485234;
_parameter_number[107] = 107; _parameter_value[107] = 0.000000000000000;  _parameter_error[107] = -999.000000000000000;
_parameter_number[108] = 108; _parameter_value[108] = 0.000000000000000;  _parameter_error[108] = -999.000000000000000;
_parameter_number[109] = 109; _parameter_value[109] = 8.044271644971598;  _parameter_error[109] = 0.360011959229178;
_parameter_number[110] = 110; _parameter_value[110] = -2.198468114762642; _parameter_error[110] = 0.044178533713445;
_parameter_number[111] = 111; _parameter_value[111] = 26.298552667260097; _parameter_error[111] = 1.615048451153438;
_parameter_number[112] = 112; _parameter_value[112] = -2.658526173320670; _parameter_error[112] = 0.053764969565157;
_parameter_number[113] = 113; _parameter_value[113] = 33.034929279837407; _parameter_error[113] = 1.795550599855257;
_parameter_number[114] = 114; _parameter_value[114] = -1.627139615013793; _parameter_error[114] = 0.055805586880306;
_parameter_number[115] = 115; _parameter_value[115] = 26.174079452540983; _parameter_error[115] = 1.282973161816828;
_parameter_number[116] = 116; _parameter_value[116] = -2.118910494514562; _parameter_error[116] = 0.049384266277363;
_parameter_number[117] = 117; _parameter_value[117] = 0.000000000000000;  _parameter_error[117] = -999.000000000000000;
_parameter_number[118] = 118; _parameter_value[118] = 0.000000000000000;  _parameter_error[118] = -999.000000000000000;
_parameter_number[119] = 119; _parameter_value[119] = 3.122415682166643;  _parameter_error[119] = 0.519917442042756;
_parameter_number[120] = 120; _parameter_value[120] = 11.139907856904129; _parameter_error[10] = 0.326290789753655;
_parameter_number[121] = 121; _parameter_value[121] = 29.146102368470210; _parameter_error[121] = 1.567192831203517;
_parameter_number[122] = 122; _parameter_value[122] = 6.631556203215280;  _parameter_error[122] = 0.464868341595447;
_parameter_number[123] = 123; _parameter_value[123] = 0.000000000000000;  _parameter_error[123] = 0.000000000000000;
_parameter_number[124] = 124; _parameter_value[124] = 7.928823290976309;  _parameter_error[124] = 0.449760935845812;
_parameter_number[125] = 125; _parameter_value[125] = 4.948420661321371;  _parameter_error[125] = 0.310436057751668;
_parameter_number[126] = 126; _parameter_value[126] = -0.053588781806890; _parameter_error[126] = 1.264956705435045;
_parameter_number[127] = 127; _parameter_value[127] = -8.455370251307063; _parameter_error[127] = 0.431374758626274;
_parameter_number[128] = 128; _parameter_value[128] = 0.000000000000000;  _parameter_error[128] = 0.000000000000000;
_parameter_number[129] = 129; _parameter_value[129] = -4.724094278696236; _parameter_error[129] = 0.370517580820305;
_parameter_number[130] = 130; _parameter_value[130] = -23.289333360304212; _parameter_error[130] = 1.695204225857380;
_parameter_number[131] = 131; _parameter_value[131] = -1.860311896516422;  _parameter_error[131] = 1.843999482073795;
_parameter_number[132] = 132; _parameter_value[132] = -13.638752211193912; _parameter_error[132] = 1.348301787723820;
_parameter_number[133] = 133; _parameter_value[133] = 0.000000000000000;  _parameter_error[133] = 0.000000000000000;
_parameter_number[134] = 134; _parameter_value[134] = -6.511009103363590; _parameter_error[134] = 0.344417203623534;
_parameter_number[135] = 135; _parameter_value[135] = -12.215597571354197; _parameter_error[135] = 1.316773691635982;
_parameter_number[136] = 136; _parameter_value[136] = -32.982507366353126; _parameter_error[136] = 1.795072160141119;
_parameter_number[137] = 137; _parameter_value[137] = -22.339804683783186; _parameter_error[137] = 1.224289757069352;
_parameter_number[138] = 138; _parameter_value[138] = 0.000000000000000; _parameter_error[138] = 0.000000000000000;
_parameter_number[139] = 139; _parameter_value[139] = -0.070000000000000; _parameter_error[139] = 0.000000000000000;
_parameter_number[140] = 140; _parameter_value[140] = 0.955319683174069; _parameter_error[140] = 0.065591431616335;
_parameter_number[141] = 141; _parameter_value[141] = 0.001737032480754; _parameter_error[141] = 0.005321783176449;
_parameter_number[142] = 142; _parameter_value[142] = 1.000000000000000; _parameter_error[142] = 0.000000000000000;
_parameter_number[143] = 143; _parameter_value[143] = -1.914503836666840; _parameter_error[143] = 0.046141357147862;
_parameter_number[144] = 144; _parameter_value[144] = 0.112673863011817; _parameter_error[144] = 0.006203024232983;
_parameter_number[145] = 145; _parameter_value[145] = -33.799002116066454; _parameter_error[145] = 1.813327539874341;
_parameter_number[146] = 146; _parameter_value[146] = 0.000000000000000; _parameter_error[146] = 0.000000000000000;
_parameter_number[147] = 147; _parameter_value[147] = 0.204240930854485; _parameter_error[147] = 0.000000000000000;
_parameter_number[148] = 148; _parameter_value[148] = 0.005351081329072; _parameter_error[148] = 0.000000000000000;
_parameter_number[149] = 149; _parameter_value[149] = 0.598710956456969; _parameter_error[149] = 0.000000000000000;
_parameter_number[150] = 150; _parameter_value[150] = 0.069852505522275; _parameter_error[150] = 0.000000000000000;
_parameter_number[151] = 151; _parameter_value[151] = 0.013034115290053; _parameter_error[151] = 0.000000000000000;
_parameter_number[152] = 152; _parameter_value[152] = 0.005413906071129; _parameter_error[152] = 0.000000000000000;
_parameter_number[153] = 153; _parameter_value[153] = 0.000730193619689; _parameter_error[153] = 0.000000000000000;
_parameter_number[154] = 154; _parameter_value[154] = 0.005452230567231; _parameter_error[154] = 0.000000000000000;
_parameter_number[155] = 155; _parameter_value[155] = 0.000142051889254; _parameter_error[155] = 0.000000000000000;
_parameter_number[156] = 156; _parameter_value[156] = 0.000084670790984; _parameter_error[156] = 0.000000000000000;
_parameter_number[157] = 157; _parameter_value[157] = 0.000000000000000; _parameter_error[157] = 0.000000000000000;
_parameter_number[158] = 158; _parameter_value[158] = 0.000387551292080; _parameter_error[158] = 0.000000000000000;
_parameter_number[159] = 159; _parameter_value[159] = 0.000000000000000; _parameter_error[159] = 0.000000000000000;
_parameter_number[160] = 160; _parameter_value[160] = 0.007567968150331; _parameter_error[160] = 0.000000000000000;
_parameter_number[161] = 161; _parameter_value[161] = 0.000000000000000; _parameter_error[161] = 0.000000000000000;
_parameter_number[162] = 162; _parameter_value[162] = 0.005914842613918; _parameter_error[162] = 0.000000000000000;
_parameter_number[163] = 163; _parameter_value[163] = 0.000000000000000; _parameter_error[163] = 0.000000000000000;
_parameter_number[164] = 164; _parameter_value[164] = 0.000000000000000; _parameter_error[164] = 0.000000000000000;
_parameter_number[165] = 165; _parameter_value[165] = 0.099456444040309; _parameter_error[165] = 0.000000000000000;
_parameter_number[166] = 166; _parameter_value[166] = 0.491700000000000; _parameter_error[166] = 0.000000000000000;
_parameter_number[167] = 167; _parameter_value[167] = 0.941830105115886; _parameter_error[167] = 0.000000000000000;
_parameter_number[168] = 168; _parameter_value[168] = 0.759845902082103; _parameter_error[168] = 0.000000000000000;
_parameter_number[169] = 169; _parameter_value[169] = 0.000000000000000; _parameter_error[169] = 0.000000000000000;
	
  /*parameter_filename = "/workfs2/bes/yuzhang1/710/StrongPhase/QCMCFilterAlg/QCMCFilterAlg-00-01-03/src/res_info.txt";

        // open parameter file and read Dalitz model
        cout << "Opening parameter file: " << parameter_filename << endl;

        FILE *pFile_fit_result;

        pFile_fit_result = fopen(parameter_filename.c_str(), "r");

        if (pFile_fit_result==NULL) {
            cout << "File error in reading parameter array." << endl;
            cout << endl;
            cout << "Can not find:  " << parameter_filename << endl;
            cout << endl;
        }

        int temp_number = -999;
        double temp_value = -999;
        double temp_error = -999;
        while (fscanf(pFile_fit_result, "%i %lf %lf\n", &temp_number, &temp_value, &temp_error) == 3 )
        {
            _parameter_number[temp_number] = temp_number;
            _parameter_value[temp_number] = temp_value;
            //_parameter_error[temp_number] = temp_error;
        }*/

        //cout << "============================================================================" << endl;
        //cout << "Constructor has read the following parameters:" << endl;

        //for (int i=0; i<NUMBER_PARAMETERS; i++) {
        //    cout << "_parameter_number[" << i <<"] == " << _parameter_number[i] << "   ";
        //    cout << "_parameter_value[" << i << "] == " << _parameter_value[i] << "   ";
        //    //cout << "_parameter_error[" << i << "] == " << _parameter_error[i] << endl;
        //}
        //cout << "============================================================================" << endl;

    //cout << "============================================================================" << endl;
    //cout << "Constructor calculates the Dalitz AmpSquared normalization integral:" << endl;
    //_DalitzNormalization = computeDalitzAmpSquaredNormalizationIntegral();
    //cout << "_DalitzNormalization == " << _DalitzNormalization << endl;
    //cout << "============================================================================" << endl;
		//
		return;
    
}

/*
 * Function compute amplitude as a function of Dalitz plot position
 */
TComplex D0ToKspipi2018::Amp_PFT(double *x) {
    TComplex total_PFT(0.0, 0.0);
    //cout<<"x[0] = "<< x[0] << " x[1] = "<<x[1]<<endl;
    total_PFT = Amp(x,_parameter_value);
    //cout<<"|A| = "<< total_PFT.Rho() << " arg(A) = "<<total_PFT.Theta()<<endl;
    return total_PFT;
}

TComplex D0ToKspipi2018::Amp(double *x, double *par) {
    double xx = x[0]; // is either msquared01 for D0 or msquared02 for D0bar
    double zz = x[1]; // is either msquared02 for D0 or msquared01 for D0bar
	if (inDalitz_01_02(xx, zz) == false) return 0;

    /*
     * Check if xx and yy are within the boundary of the Dalitz-plot phasespace.
     *
     * If not, return 0.
     */


    double yy = mD0 * mD0 + mKs * mKs + 2 * mPi * mPi - xx - zz; // compute msquared12 to be used in amplitude calculation below

    /*
     * The Dalitz fit has been performed in cartesian coordinates.
     * For the computation of the of the resonance amplitudes convert the read-in cartesian parameters into polar ones.
     */

    TComplex TComplex_omega( par[omega_realpart], par[omega_imaginarypart], kFALSE);
    TComplex TComplex_Kstar892minus( par[Kstar892minus_realpart], par[Kstar892minus_imaginarypart], kFALSE);
    TComplex TComplex_Kstarzero1430minus( par[Kstarzero1430minus_realpart], par[Kstarzero1430minus_imaginarypart], kFALSE);
    TComplex TComplex_Kstartwo1430minus( par[Kstartwo1430minus_realpart], par[Kstartwo1430minus_imaginarypart], kFALSE);
    TComplex TComplex_Kstar1680minus( par[Kstar1680minus_realpart], par[Kstar1680minus_imaginarypart], kFALSE);
    TComplex TComplex_Kstar1410minus( par[Kstar1410minus_realpart], par[Kstar1410minus_imaginarypart], kFALSE);
    TComplex TComplex_Kstar892plus( par[Kstar892plus_realpart], par[Kstar892plus_imaginarypart], kFALSE);
    TComplex TComplex_Kstarzero1430plus( par[Kstarzero1430plus_realpart], par[Kstarzero1430plus_imaginarypart], kFALSE);
    TComplex TComplex_Kstartwo1430plus( par[Kstartwo1430plus_realpart], par[Kstartwo1430plus_imaginarypart], kFALSE);
    TComplex TComplex_Kstar1410plus( par[Kstar1410plus_realpart], par[Kstar1410plus_imaginarypart], kFALSE);
    TComplex TComplex_ftwo1270( par[ftwo1270_realpart], par[ftwo1270_imaginarypart], kFALSE);
    TComplex TComplex_rho1450( par[rho1450_realpart], par[rho1450_imaginarypart], kFALSE);

    TComplex TComplex_Kmatrix_beta1( par[Kmatrix_beta1_realpart], par[Kmatrix_beta1_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_beta2( par[Kmatrix_beta2_realpart], par[Kmatrix_beta2_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_beta3( par[Kmatrix_beta3_realpart], par[Kmatrix_beta3_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_beta4( par[Kmatrix_beta4_realpart], par[Kmatrix_beta4_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_beta5( par[Kmatrix_beta5_realpart], par[Kmatrix_beta5_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_f_prod_11( par[Kmatrix_f_prod_11_realpart], par[Kmatrix_f_prod_11_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_f_prod_12( par[Kmatrix_f_prod_12_realpart], par[Kmatrix_f_prod_12_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_f_prod_13( par[Kmatrix_f_prod_13_realpart], par[Kmatrix_f_prod_13_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_f_prod_14( par[Kmatrix_f_prod_14_realpart], par[Kmatrix_f_prod_14_imaginarypart], kFALSE);
    TComplex TComplex_Kmatrix_f_prod_15( par[Kmatrix_f_prod_15_realpart], par[Kmatrix_f_prod_15_imaginarypart], kFALSE);

    /*
     * Construct the resonances.
     * For Breit-Wigner (BW) resonances the constructor parameters are: "BW", amplitude, phase, mass, width, spin, resonance radius (Blatt-Weisskopf), D meson radius (Blatt-Weisskopf), resonance orientation (X=Kpi(right sign), Y=pipi, Z=Kpi(wrong sign)
     * Amplitudes and phases are measured relative to the rho(770), which is set to amplitude=1 and phase=0
     */

    // pipi resonances (L!=0)
    Resonance ResRho("BW", 1, 0, par[rho770_Mass], par[rho770_Width], 1, 1.5, 5.0, "Y");
    Resonance ResOmega("BW", TComplex_omega.Rho(), TComplex_omega.Theta() * 180. / TMath::Pi(), par[omega_Mass], par[omega_Width], 1, 1.5, 5.0, "Y");
    Resonance Resf2_1270("BW", TComplex_ftwo1270.Rho(), TComplex_ftwo1270.Theta() * 180. / TMath::Pi(), par[ftwo1270_Mass], par[ftwo1270_Width], 2, 1.5, 5.0, "Y");
    Resonance ResRho_1450("BW", TComplex_rho1450.Rho(), TComplex_rho1450.Theta() * 180. / TMath::Pi(), par[rho1450_Mass], par[rho1450_Width], 1, 1.5, 5.0, "Y");

    // K* resonances, Cabibbo-favored
    Resonance ResKstar("BW", TComplex_Kstar892minus.Rho(), TComplex_Kstar892minus.Theta() * 180. / TMath::Pi(), par[Kstar892_Mass], par[Kstar892_Width], 1, 1.5, 5.0, "X");

    Resonance ResKstar0_1430("LASS", TComplex_Kstarzero1430minus.Rho(), TComplex_Kstarzero1430minus.Theta() * 180. / TMath::Pi(), par[Kstarzero1430_Mass], par[Kstarzero1430_Width], 0, 1.5, 5.0, "X",
            par[LASS_F],
            par[LASS_phi_F],
            par[LASS_R],
            par[LASS_phi_R],
            par[LASS_a],
            par[LASS_r]);

    Resonance ResKstar2_1430("BW", TComplex_Kstartwo1430minus.Rho(), TComplex_Kstartwo1430minus.Theta() * 180. / TMath::Pi(), par[Kstartwo1430_Mass], par[Kstartwo1430_Width], 2, 1.5, 5.0, "X");
    Resonance ResKstar_1680("BW", TComplex_Kstar1680minus.Rho(), TComplex_Kstar1680minus.Theta() * 180. / TMath::Pi(), par[Kstar1680_Mass], par[Kstar1680_Width], 1, 1.5, 5.0, "X");
    Resonance ResKstar_1410("BW", TComplex_Kstar1410minus.Rho(), TComplex_Kstar1410minus.Theta() * 180. / TMath::Pi(), par[Kstar1410_Mass], par[Kstar1410_Width], 1, 1.5, 5.0, "X");

    // K* resonances, doubly Cabibbo-suppressed
    Resonance ResKstar_DCS("BW", TComplex_Kstar892plus.Rho(), TComplex_Kstar892plus.Theta() * 180. / TMath::Pi(), par[Kstar892_Mass], par[Kstar892_Width], 1, 1.5, 5.0, "Z");

    Resonance ResKstar0_1430_DCS("LASS", TComplex_Kstarzero1430plus.Rho(), TComplex_Kstarzero1430plus.Theta() * 180. / TMath::Pi(), par[Kstarzero1430_Mass], par[Kstarzero1430_Width], 0, 1.5, 5.0, "Z",
            par[LASS_F],
            par[LASS_phi_F],
            par[LASS_R],
            par[LASS_phi_R],
            par[LASS_a],
            par[LASS_r]);

    Resonance ResKstar2_1430_DCS("BW", TComplex_Kstartwo1430plus.Rho(), TComplex_Kstartwo1430plus.Theta() * 180. / TMath::Pi(), par[Kstartwo1430_Mass], par[Kstartwo1430_Width], 2, 1.5, 5.0, "Z");
    Resonance ResKstar_1410_DCS("BW", TComplex_Kstar1410plus.Rho(), TComplex_Kstar1410plus.Theta() * 180. / TMath::Pi(), par[Kstar1410_Mass], par[Kstar1410_Width], 1, 1.5, 5.0, "Z");

    // K-matrix for the pipi S-wave
    Resonance ResKMatrix("KMatrix", "Y",
            TComplex_Kmatrix_beta1.Rho(),
            TComplex_Kmatrix_beta1.Theta(),
            TComplex_Kmatrix_beta2.Rho(),
            TComplex_Kmatrix_beta2.Theta(),
            TComplex_Kmatrix_beta3.Rho(),
            TComplex_Kmatrix_beta3.Theta(),
            TComplex_Kmatrix_beta4.Rho(),
            TComplex_Kmatrix_beta4.Theta(),
            TComplex_Kmatrix_beta5.Rho(),
            TComplex_Kmatrix_beta5.Theta(),
            TComplex_Kmatrix_f_prod_11.Rho(),
            TComplex_Kmatrix_f_prod_11.Theta(),
            TComplex_Kmatrix_f_prod_12.Rho(),
            TComplex_Kmatrix_f_prod_12.Theta(),
            TComplex_Kmatrix_f_prod_13.Rho(),
            TComplex_Kmatrix_f_prod_13.Theta(),
            TComplex_Kmatrix_f_prod_14.Rho(),
            TComplex_Kmatrix_f_prod_14.Theta(),
            TComplex_Kmatrix_f_prod_15.Rho(),
            TComplex_Kmatrix_f_prod_15.Theta(),
            par[Kmatrix_s_prod_0]
    );

/*
 * Compute the total amplitude by the coherent sum of the individual amplitudes.
 *
 * Resonance.contribution(xx, yy) returns the amplitude and phase of individual resonances as a function of the Dalitz plot position xx and yy.
 */
    TComplex total_amp(0.0, 0.0);

    total_amp +=
        ResRho.contribution(xx, yy) +
        ResOmega.contribution(xx, yy) +
        ResKstar.contribution(xx, yy) +
        ResKstar0_1430.contribution(xx, yy) +
        ResKstar2_1430.contribution(xx, yy) +
        ResKstar_1680.contribution(xx, yy) +
        ResKstar_1410.contribution(xx, yy) +
        ResKstar_DCS.contribution(xx, yy) +
        ResKstar0_1430_DCS.contribution(xx, yy) +
        ResKstar2_1430_DCS.contribution(xx, yy) +
        ResKstar_1410_DCS.contribution(xx, yy) +
        Resf2_1270.contribution(xx, yy) +
        ResRho_1450.contribution(xx, yy) +
        ResKMatrix.contribution(xx, yy);

    /*
     * Return the total amplitude
     */
    return total_amp;
}

/*
 * Function to determine, if within the Dalitz plot phase space (x: Kpi right-sign, y: Kpi wrong-sign coordinates)
 */

bool D0ToKspipi2018::inDalitz_01_02(Double_t x, Double_t y) {
    static const Double_t PI_mass = 0.13957017;
    static const Double_t K0_mass = 0.49761401;
    static const Double_t mD0 = 1.8648399;

    double msquared01 = x; // Kpi(RS)
    double msquared02 = y; // Kpi(WS)
    double msquared12 = mD0*mD0 + K0_mass*K0_mass + 2*PI_mass*PI_mass - msquared01 - msquared02;

    Double_t local_ma = K0_mass;
    Double_t local_mb = PI_mass;
    Double_t local_mc = PI_mass;

    Double_t local_xmin = pow(local_ma + local_mb,2);
    Double_t local_xmax = pow(mD0 - local_mc,2);

    // Find energy of b(c) in ab frame
    Double_t ebab = (x - local_ma*local_ma + local_mb*local_mb)/(2.0*sqrt(x));
    Double_t ecab = (mD0*mD0 - x - local_mc*local_mc)/(2.0*sqrt(x));

    Double_t yhi = pow(ebab+ecab,2) - pow( sqrt(ebab*ebab-local_mb*local_mb)-sqrt(ecab*ecab-local_mc*local_mc) ,2);
    Double_t ylo = pow(ebab+ecab,2) - pow( sqrt(ebab*ebab-local_mb*local_mb)+sqrt(ecab*ecab-local_mc*local_mc) ,2);

    // Initialize boolean variable as false.
    bool inDal = false;

    // Return true, if within the Dalitz-plot phase space.
    if ((local_xmin <= x) && (x <= local_xmax) && (ylo <= msquared12) && (msquared12 <= yhi)) { inDal = true; }

    return inDal;
}

/*
 * Function to integrate the magnitude-squared of the complex amplitude over the Dalitz-plot phase-space.
 *
 * The integration is performed in a very basic algorithm: A double loop over the Dalitz-plot coordinates performs the sum of the kind: \sum_{x,y} Amp_squared(x, y) * dx * dy
 */

/*double D0ToKspipi2018::computeDalitzAmpSquaredNormalizationIntegral() {

    int nSteps = 5; 

    const double mD0 = 1.8648399;
    const double mKs = 0.49761401;
    const double mPi = 0.13957017;
    double MiniX = (mKs + mPi) * (mKs + mPi);
    double MaxiX = (mD0 - mPi) * (mD0 - mPi);
    double hX = (MaxiX - MiniX) / nSteps;
    double MiniY = (mKs + mPi) * (mKs + mPi);
    double MaxiY = (mD0 - mPi) * (mD0 - mPi);
    double hY = (MaxiY - MiniY) / nSteps;

    double D0Sum[10], D0barSum[10];
    TComplex D0D0barSum[10];
    int points[10];

    double sum = 0;
    for (int i = 0; i < nSteps; i++) {
        for (int j = 0; j < nSteps; j++) {
            double xx[2];
            xx[0] = MiniX + (i + 0.5) * hX;
            xx[1] = MiniY + (j + 0.5) * hY;
            double xxb[2];
            xxb[0] = xx[1];
            xxb[1] = xx[0];
            if(!inDalitz_01_02(xx[0],xx[1])) continue;
            TComplex temp_Amp = Amp(xx, _parameter_value);
            TComplex temp_Ampb = Amp(xxb, _parameter_value);;
            double temp_AmpSquared = temp_Amp * TComplex::Conjugate(temp_Amp);
            sum += hX * hY * temp_AmpSquared;
            if (xx[0]<=xx[1]) {
                int bin = getBin(xx);
                if (bin == -1) bin = 8;
                ++points[ bin];
                D0Sum[bin] += (temp_Amp).Rho();
                D0barSum[bin] += (temp_Ampb).Rho();
                D0D0barSum[bin] += temp_Amp * TComplex::Conjugate( temp_Ampb );
            }
        }
    }

    return sum;
}*/

/*int D0ToKspipi2018::getBin(double *xx){
    double deltaD = GetStrongPhase(xx);
    int thisbin = -1;
    for (int bin=0; bin<8; bin++){
        if((deltaD >= (bin-0.5)*2*TMath::Pi()/8) && (deltaD < (bin+0.5)*2*TMath::Pi()/8)) thisbin = bin;
    }
    return thisbin;
}*/


/*double D0ToKspipi2018::GetStrongPhase(double *x){
	EvtVector4R kl = _p4CM[0];
	EvtVector4R pip = _p4CM[1];
	EvtVector4R pim = _p4CM[2];
	EvtVector4R klb(kl.get(0),-1.*kl.get(1),-1.*kl.get(2),-1.*kl.get(3));
	EvtVector4R pipb(pim.get(0),-1.*pim.get(1),-1.*pim.get(2),-1.*pim.get(3));
	EvtVector4R pimb(pip.get(0),-1.*pip.get(1),-1.*pip.get(2),-1.*pip.get(3));
	double xb[2];
	xb[0]=(klb+pimb).mass2();
	xb[1]=(klb+pipb).mass2();
	TComplex amp = Amp(x, _parameter_value);
	TComplex ampb = Amp(xb,_parameter_value);
	double temp = amp.Theta()-ampb.Theta();
	while (temp < -TMath::Pi()){
		temp += 2.0*TMath::Pi();
	}
	while (temp > -TMath::Pi()){
		temp -= 2.0*TMath::Pi();
	}
	return temp;
}*/
