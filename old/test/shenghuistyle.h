#ifndef _SHENGHUISTYLE_H_
#define _SHENGHUISTYLE_H_
// all users - please change the name of this file to lhcbStyle.C
// Commits to lhcbdocs svn of .C files are not allowed
void lhcbStyle()
{

  // define names for colours
  Int_t black  = 1;
  Int_t red    = 2;
  Int_t green  = 3;
  Int_t blue   = 4;
  Int_t magenta= 7;
  Int_t cyan   = 8;
  Int_t purple = 9;
  Int_t yellow = 5;

////////////////////////////////////////////////////////////////////
// PURPOSE:
//
// This macro defines a standard style for (black-and-white)
// "publication quality" LHCb ROOT plots.
//
// USAGE:
//
// Include the lines
//   gROOT->ProcessLine(".L lhcbstyle.C");
//   lhcbStyle();
// at the beginning of your root macro.
//
// Example usage is given in myPlot.C
//
// COMMENTS:
//
// Font:
//
// The font is chosen to be 132, this is Times New Roman (like the text of
//  your document) with precision 2.
//
// "Landscape histograms":
//
// The style here is designed for more or less square plots.
// For longer histograms, or canvas with many pads, adjustements are needed.
// For instance, for a canvas with 1x5 histograms:
//  TCanvas* c1 = new TCanvas("c1", "L0 muons", 600, 800);
//  c1->Divide(1,5);
//  Adaptions like the following will be needed:
//  gStyle->SetTickLength(0.05,"x");
//  gStyle->SetTickLength(0.01,"y");
//  gStyle->SetLabelSize(0.15,"x");
//  gStyle->SetLabelSize(0.1,"y");
//  gStyle->SetStatW(0.15);
//  gStyle->SetStatH(0.5);
//
// Authors: Thomas Schietinger, Andrew Powell, Chris Parkes, Niels Tuning
// Maintained by Editorial board member (currently Niels)
///////////////////////////////////////////////////////////////////

  // Use times new roman, precision 2
  Int_t lhcbFont        = 132;  // Old LHCb style: 62;
  // Line thickness
  Double_t lhcbWidth    = 2.00; // Old LHCb style: 3.00;
  // Text size
  Double_t lhcbTSize    = 0.06;

  // use plain black on white colors
  gROOT->SetStyle("Plain");
  TStyle *lhcbStyle= new TStyle("lhcbStyle","LHCb plots style");

  lhcbStyle->SetErrorX(0); //  don't suppress the error bar along X

  lhcbStyle->SetFillColor(1);
  lhcbStyle->SetFillStyle(1001);   // solid
  lhcbStyle->SetFrameFillColor(0);
  lhcbStyle->SetFrameBorderMode(0);
  lhcbStyle->SetPadBorderMode(0);
  lhcbStyle->SetPadColor(0);
  lhcbStyle->SetCanvasBorderMode(0);
  lhcbStyle->SetCanvasColor(0);
  lhcbStyle->SetStatColor(0);
  lhcbStyle->SetLegendBorderSize(0);

  // If you want the usual gradient palette (blue -> red)
  lhcbStyle->SetPalette(56);//56 golden 75 cherry
  // If you want colors that correspond to gray scale in black and white:
  int colors[8] = {9,4,7,3,3,6,5,2};
//  lhcbStyle->SetPalette(8,colors);

  // set the paper & margin sizes
  lhcbStyle->SetPaperSize(20,26);
  lhcbStyle->SetPadTopMargin(0.05);
  lhcbStyle->SetPadRightMargin(0.05); // increase for colz plots
  lhcbStyle->SetPadBottomMargin(0.16);
  lhcbStyle->SetPadLeftMargin(0.14);

  // use large fonts
  lhcbStyle->SetTextFont(lhcbFont);
  lhcbStyle->SetTextSize(lhcbTSize);
  lhcbStyle->SetLabelFont(lhcbFont,"x");
  lhcbStyle->SetLabelFont(lhcbFont,"y");
  lhcbStyle->SetLabelFont(lhcbFont,"z");
  lhcbStyle->SetLabelSize(lhcbTSize,"x");
  lhcbStyle->SetLabelSize(lhcbTSize,"y");
  lhcbStyle->SetLabelSize(lhcbTSize,"z");
  lhcbStyle->SetTitleFont(lhcbFont);
  lhcbStyle->SetTitleFont(lhcbFont,"x");
  lhcbStyle->SetTitleFont(lhcbFont,"y");
  lhcbStyle->SetTitleFont(lhcbFont,"z");
  lhcbStyle->SetTitleSize(1.2*lhcbTSize,"x");
  lhcbStyle->SetTitleSize(1.2*lhcbTSize,"y");
  lhcbStyle->SetTitleSize(1.2*lhcbTSize,"z");

  // use medium bold lines and thick markers
  lhcbStyle->SetLineWidth(lhcbWidth);
  lhcbStyle->SetFrameLineWidth(lhcbWidth);
  lhcbStyle->SetHistLineWidth(lhcbWidth);
  lhcbStyle->SetFuncWidth(lhcbWidth);
  lhcbStyle->SetGridWidth(lhcbWidth);
  lhcbStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes
  lhcbStyle->SetMarkerStyle(20);
//  lhcbStyle->SetMarkerSize(0.5);
  lhcbStyle->SetMarkerSize(1.0);

  // label offsets
  lhcbStyle->SetLabelOffset(0.010,"X");
  lhcbStyle->SetLabelOffset(0.010,"Y");

  // by default, do not display histogram decorations:
  lhcbStyle->SetOptStat(0);
  //lhcbStyle->SetOptStat("emr");  // show only nent -e , mean - m , rms -r
  // full opts at http://root.cern.ch/root/html/TStyle.html#TStyle:SetOptStat
  lhcbStyle->SetStatFormat("6.3g"); // specified as c printf options
  lhcbStyle->SetOptTitle(0);
 // lhcbStyle->SetOptFit(0);
  //lhcbStyle->SetOptFit(1011); // order is probability, Chi2, errors, parameters
  lhcbStyle->SetOptFit(1111); // order is probability, Chi2, errors, parameters
  //titles
  lhcbStyle->SetTitleOffset(0.95,"X");
  lhcbStyle->SetTitleOffset(0.95,"Y");
  lhcbStyle->SetTitleOffset(1.2,"Z");
  lhcbStyle->SetTitleFillColor(0);
  lhcbStyle->SetTitleStyle(0);
  lhcbStyle->SetTitleBorderSize(0);
  lhcbStyle->SetTitleFont(lhcbFont,"title");
  lhcbStyle->SetTitleX(0.0);
  lhcbStyle->SetTitleY(1.0);
  lhcbStyle->SetTitleW(1.0);
  lhcbStyle->SetTitleH(0.05);

  // look of the statistics box:
  lhcbStyle->SetStatBorderSize(0);
  lhcbStyle->SetStatFont(lhcbFont);
  lhcbStyle->SetStatFontSize(0.05);
  lhcbStyle->SetStatX(0.9);
  lhcbStyle->SetStatY(0.9);
  lhcbStyle->SetStatW(0.25);
  lhcbStyle->SetStatH(0.15);

  // put tick marks on top and RHS of plots
  lhcbStyle->SetPadTickX(1);
  lhcbStyle->SetPadTickY(1);

  // histogram divisions: only 5 in x to avoid label overlaps
  lhcbStyle->SetNdivisions(505,"x");
  lhcbStyle->SetNdivisions(510,"y");

  gROOT->SetStyle("lhcbStyle");
  gROOT->ForceStyle();

  // add LHCb label
  TPaveText *lhcbName = new TPaveText(gStyle->GetPadLeftMargin() + 0.05,
                           0.87 - gStyle->GetPadTopMargin(),
                           gStyle->GetPadLeftMargin() + 0.20,
                           0.95 - gStyle->GetPadTopMargin(),
                           "BRNDC");
  lhcbName->AddText("LHCb");
  lhcbName->SetFillColor(0);
  lhcbName->SetTextAlign(12);
  lhcbName->SetBorderSize(0);

  TText *lhcbLabel = new TText();
  lhcbLabel->SetTextFont(lhcbFont);
  lhcbLabel->SetTextColor(1);
  lhcbLabel->SetTextSize(lhcbTSize);
  lhcbLabel->SetTextAlign(12);

  TLatex *lhcbLatex = new TLatex();
  lhcbLatex->SetTextFont(lhcbFont);
  lhcbLatex->SetTextColor(1);
  lhcbLatex->SetTextSize(lhcbTSize);
  lhcbLatex->SetTextAlign(12);

}
void SetStyle(){
  // No Canvas Border
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasBorderSize(0);
  // White BG
  gStyle->SetCanvasColor(10);
  // Format for axes
  gStyle->SetLabelFont(42,"xyz");
  gStyle->SetLabelSize(0.06,"xyz");
  gStyle->SetLabelOffset(0.01,"xyz");
  gStyle->SetNdivisions(505,"xyz");
  gStyle->SetTitleFont(42,"xyz");
  gStyle->SetTitleColor(1,"xyz");
  gStyle->SetTitleSize(0.07,"xyz");
  gStyle->SetTitleOffset(1.15,"xyz");
  // No pad borders
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadBorderSize(0);
  // White BG
  gStyle->SetPadColor(10);
  // Margins for labels etc.
  gStyle->SetPadLeftMargin(0.17);
  gStyle->SetPadBottomMargin(0.17);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadTopMargin(0.05);
  // No error bars in x direction
  gStyle->SetErrorX(0);
  // Format legend
  gStyle->SetLegendBorderSize(0);
  	gStyle->SetOptDate(0);
	gStyle->SetOptStat(0);
	gStyle->SetOptFit(0);
	gStyle->SetOptTitle(0);
	gStyle->SetOptTitle(kFALSE);
	gStyle->SetLineWidth(2);
	//gStyle->SetLegendSize(0.05);
}

void shenghuistyle()
{
	gROOT->Reset();
	gROOT->SetStyle("Pub");
	gStyle->SetPalette(56);
	gStyle->SetCanvasBorderMode(0);
	gStyle->SetCanvasBorderSize(0);
	gStyle->SetCanvasColor(10);

	gStyle->SetTextSize(0.06);
	gStyle->SetLabelFont(42,"xyz");
	gStyle->SetLabelSize(0.06,"xyz");
	gStyle->SetLabelOffset(0.01,"xyz");
	gStyle->SetLabelColor(1,"xyz");
  	gStyle->SetNdivisions(505,"xyz");


	gStyle->SetStripDecimals(kFALSE);
	gStyle->SetTitleFont(42,"xyz");
	gStyle->SetTitleColor(1,"xyz");
	gStyle->SetTitleSize(0.07,"xyz");
	gStyle->SetTitleOffset(1.15,"xyz");//distance between title and graph


	gStyle->SetPadBorderMode(0);
	gStyle->SetPadBorderSize(0.0);
	gStyle->SetPadLeftMargin(0.17);
  	gStyle->SetPadBottomMargin(0.17);
  	gStyle->SetPadRightMargin(0.05);
  	gStyle->SetPadTopMargin(0.05);

	gStyle->SetLegendBorderSize(0);

	gStyle->SetOptDate(0);
	gStyle->SetOptStat(0);
	gStyle->SetOptFit(0);
	gStyle->SetOptTitle(0);
	gStyle->SetOptTitle(kFALSE);

	// No error bars in x direction
	gStyle->SetErrorX(0);
	// Format legend
	gStyle->SetLegendBorderSize(0);

	gStyle->SetLineWidth(2);
	gStyle->SetMarkerSize(1);
	gStyle->SetMarkerStyle(20);
	gStyle->SetMarkerColor(1);
	//gPad->SetLogy(1);

}

void Trk(Double_t &p1, Int_t &tag, Double_t &weight)
{
	double k_f[11];
	double kaonp[11] = {-0.0150,0.0263,0.0001,-0.0015,0.0029,-0.0064,0.0068,0.0016,-0.0066,-0.0070,0.0013};
	double kaonm[11] = {-0.0472,0.0317,0.0098,0.0045,-0.0016,-0.0002,0.0022,0.0004,0.0113,-0.0044,-0.0026};
	double pionp[11] = {-0.002,-0.0007,0.0013,0.0023,0.0031,0.0071,0.0033,-0.0032,0.0046,0.0015,0.0021};
	double pionm[11] = {-0.0265,0.007,-0.0056,0.0041,0.0031,0.0015,0.0018,0.0026,0.0006,0.0037,0.0009};
	/*	switch(tag){
		case 1:
		k_f[11] = kaonp[11];
		break;
		case -1:
		k_f[11] = kaonm[11];
		break;
		case 2:
		k_f[11] = pionp[11];
		break;
		case -2:
		k_f[11] = pionm[11];
		break;
		default:
		cout<<"Invalid tag"<<endl;
		break;
		}*/
	if(tag==1){
		for (int i=0; i<11; i++) {
			k_f[i] = kaonp[i];
		}
	}

	if(tag==-1){
		for (int i=0; i<11; i++) {
			k_f[i] = kaonm[i];
		}
	}
	if(tag==2){
		for (int i=0; i<11; i++) {
			k_f[i] = pionp[i];
		}
	}

	if(tag==-2){
		for (int i=0; i<11; i++) {
			k_f[i] = pionm[i];
		}
	}

	/*	if(tag==1){ k_f[11]={-0.0150,0.0263,0.0001,-0.0015,0.0029,-0.0064,0.0068,0.0016,-0.0066,-0.0070,0.0013};}
		if(tag==-1){ k_f[11]={-0.0472,0.0317,0.0098,0.0045,-0.0016,-0.0002,0.0022,0.0004,0.0113,-0.0044,-0.0026};}
		if(tag==2){k_f[11]={-0.002,-0.0007,0.0013,0.0023,0.0031,0.0071,0.0033,-0.0032,0.0046,0.0015,0.0021};}
		if(tag==-2){k_f[11]={-0.0265,0.007,-0.0056,0.0041,0.0031,0.0015,0.0018,0.0026,0.0006,0.0037,0.0009};}*/
	if(p1<0.2)
	{weight=1+k_f[0];}
	else if(p1<0.3)
	{weight=1+k_f[1];}
	else if(p1<0.4)
	{weight=1+k_f[2];}
	else if(p1<0.5)
	{weight=1+k_f[3];}
	else if(p1<0.6)
	{weight=1+k_f[4];}
	else if(p1<0.7)
	{weight=1+k_f[5];}
	else if(p1<0.8)
	{weight=1+k_f[6];}
	else if(p1<0.9)
	{weight=1+k_f[7];}
	else if(p1<1.0)
	{weight=1+k_f[8];}
	else if(p1<1.1)
	{weight=1+k_f[9];}
	else if(p1<1.2)
	{weight=1+k_f[10];}

}
void PID(Double_t &p1, Int_t &tag, Double_t &weight)
{
	double k_f[5];
	double kaon[5] = {-0.0554,-0.0086,-0.0045,-0.0042,0.0038};
	double pion[5] = {-0.0123,0.00,-0.001,-0.0046,-0.0022};
	if(tag==1){
		for (int i=0; i<5; i++) {
			k_f[i] = kaon[i];
		}
	}

	if(tag==0){     
		for (int i=0; i<5; i++) {
			k_f[i] = pion[i];
		}
	}
	if(p1<0.2)
	{weight=1+k_f[0];}
	else if(p1<0.4)
	{weight=1+k_f[1];}
	else if(p1<0.6)
	{weight=1+k_f[2];}
	else if(p1<0.8)
	{weight=1+k_f[3];}
	else if(p1<1.0)
	{weight=1+k_f[4];}
}

void getChi2(TH1D *histdata, TH1D *histmc, Int_t *ndf, double *fval)
{
	int nbinXdata = histdata->GetNbinsX(); 

	double chi2 = 0; 
	int ndf_temp = 0;
	double tmp;
	for (int ix = 1; ix <= nbinXdata; ++ix) { 
		if ( histdata->GetBinError(ix) > 0 ) { 
			tmp = (histdata->GetBinContent(ix) - histmc->GetBinContent(ix))/histdata->GetBinError(ix);
			chi2 += tmp*tmp; 
			ndf_temp++;
		} 
		*ndf = ndf_temp;
		*fval = chi2; 
	}
}

void process (Long64_t &nentries, Long64_t &i)
{
	if(i==Int_t(nentries*0.1)) cout<<"*******************************completed 00%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.1)) cout<<"*******************************completed 10%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.2)) cout<<"*******************************completed 20%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.3)) cout<<"*******************************completed 30%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.4)) cout<<"*******************************completed 40%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.5)) cout<<"*******************************completed 50%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.6)) cout<<"*******************************completed 60%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.7)) cout<<"*******************************completed 70%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.8)) cout<<"*******************************completed 80%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*0.9)) cout<<"*******************************completed 90%"<<"**************************************"<<endl;
	if(i==Int_t(nentries*1-1)) cout<<"*******************************completed !!!"<<"**************************************"<<endl;

	//	printf("\033[2J"); // Clears the console
	//	printf("Loading: [%-50.*s] %d%%", i/2, "===================================================", i);

	// Sleep for a short time to simulate loading
	//usleep(50000);
}
void getmass2(Double_t P1[4], Double_t P2[4] , Double_t &M_AB)
{
	TLorentzVector A_P4(P1);
	TLorentzVector B_P4(P2);
	M_AB = (A_P4+B_P4).Mag();
}
void getmass3(Double_t P1[4], Double_t P2[4], Double_t P3[4], Double_t &M_ABC)
{
	TLorentzVector A_P4(P1);
	TLorentzVector B_P4(P2);
	TLorentzVector C_P4(P3);
	M_ABC = (A_P4+B_P4+C_P4).Mag();
}
void getp(Double_t P1[4], Double_t &P)
{
	TLorentzVector A_P4(P1);
	P=A_P4.Rho();
}
void getcos(Double_t P1[4], Double_t &cos)
{
	TLorentzVector A_P4(P1);
	cos=A_P4.CosTheta();
}

double cal_eff_err(double N_sig, double N_tot) {
	double delta_N_sig = std::sqrt(N_sig);
	double delta_N_tot = std::sqrt(N_tot);

	double d_eff_d_N_sig = 1.0 / N_tot;
	double d_eff_d_N_tot = -N_sig / (N_tot * N_tot);

	double delta_eff = std::sqrt(
			std::pow(d_eff_d_N_sig * delta_N_sig, 2) +
			std::pow(d_eff_d_N_tot * delta_N_tot, 2)
			);

	return delta_eff;
}
// Write "BESIII" in the upper right corner
void WriteBes3(){
	TLatex * bes3 = new TLatex(0.84,0.84, "BESIII");
	bes3->SetNDC();
	bes3->SetTextFont(72);
	bes3->SetTextSize(0.1);
	bes3->SetTextAlign(33);
	bes3->Draw();
}
void Writedataset(){
	TLatex * dataset = new TLatex(0.87,0.87, "7.9 fb^{-1}");
	dataset->SetNDC();
	dataset->SetTextFont(132);
	dataset->SetTextSize(0.08);
	dataset->SetTextAlign(33);
	dataset->Draw();
}
void Writesqrt(int i){
	double energy[13] = {4.600, 4.612,4.628,4.641,4.661,4.682,4.699,4.740,4.750,4.781,4.843,4.918,4.951};
	TLatex * dataset = new TLatex(0.94,0.94, Form("#sqrt{s} = %.3f GeV",energy[i]));
	dataset->SetNDC();
	dataset->SetTextFont(72);
	dataset->SetTextSize(0.05);
	dataset->SetTextAlign(33);
	dataset->Draw();
}
void Write(TString str){
        TLatex * dataset = new TLatex(0.60,0.63, Form("%s",str.Data()));//0.78 for dp
        dataset->SetNDC();
        dataset->SetTextFont(42);
        dataset->SetTextSize(0.05);
        dataset->SetTextAlign(11);
        dataset->Draw();
}

//             // Write "Preliminary" below BESIII -
//             // to be used together with WriteBes3()
void WritePreliminary(){
	TLatex * prelim = new TLatex(0.94,0.86, "Preliminary");
	prelim->SetNDC();
	prelim->SetTextFont(62);
	prelim->SetTextSize(0.055);
	prelim->SetTextAlign(33);
	prelim->Draw();
}
double ratio_err(double sig_eff, double sig_eff_err, double ref_eff, double ref_eff_err) {
	double ratio = sig_eff / ref_eff;
	double d_ratio_sig_eff = 1 / ref_eff;
	double d_ratio_ref_eff = - sig_eff/pow(ref_eff,2); 
	//    double ratio_err = ratio * sqrt(pow(sig_eff_err/sig_eff, 2) + pow(ref_eff_err/ref_eff, 2));
	double ratio_err = sqrt(
			pow(d_ratio_sig_eff*sig_eff_err,2) +
			pow(d_ratio_ref_eff*ref_eff_err,2) 
			);
	return ratio_err;
}



#endif
