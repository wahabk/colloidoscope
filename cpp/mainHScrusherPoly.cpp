
#include <iostream>
//#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <time.h>
#include <math.h>

using namespace std;

#define PI 3.141592654

// paddy royall may 2006
// towards portablity

#define NPART       	512 // 2048
#define ETA         	0.28
#define ETAFINAL	    0.51
#define ETADIFF	        0.001

#define TMAX            10                       // # timesteps to equilibrate  units of 1/100 tauB
#define TTIMES 	        10                       // no of iterations printed out
#define FRAMES      	100                      // number of frames for movie and output files
#define GFRAMES      	1000                     // number g of r samples
#define NEIGHBOURS  	256                      // no of neighbours
#define POLYDISP     	0.0//0.04//5                   // polydispersity
#define STEPRANGE     	0.003//4                  // range of step in potential
#define STEPHEIGHT     	100.0//4                 // height of step in kt
#define PRINTFILE  	1                        // control output of xyz files

double rCut = 2.0;                    // interaction range
const double rho  = ETA * 6.0 / PI;               // number density
double sidex, sidey, sidez;                       // side
double invSidex, invSidey, invSidez;              // invSide
double /*ratio,*/ step=0.1;
long unsigned int accepted=0, rejected=0;         // MC acceptance control
int bin;                                          // number of bins in g(r)

///////////////////////////////////////////////////////////////////////////////////////////
//
//      R A N D O M
//
///////////////////////////////////////////////////////////////////////////////////////////

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.8e-15
#define RNMX (1.0-EPS)

static long idum;

//Long period (> 2 * 10^18 ) random number generator of L'Ecuyer with Bays-Durham shuffle
//and added safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of
//the endpoint values). Call with idum a negative integer to initialize; thereafter, do not alter
//idum between successive deviates in a sequence. RNMX should approximate the largest floating
//value that is less than 1.

void sran2(long sr)
{
	 if(sr > 0) sr = -sr;
	 idum = sr;
}

double ran2(void)
{
	 int j;
	 long k;
	 static long idum2 = 123456789;
	 static long iy = 0;
	 static long iv[NTAB];
	 double temp;

	 if (idum <= 0) {             //Initialize.
			if (-idum < 1) idum = 1; //Be sure to prevent idum = 0.
			else idum = -idum;
			idum2 = idum;
			for (j = NTAB + 7; j >= 0; j--){   // Load the shuffle table (after 8 warm-ups).
				 k= idum / IQ1;
				 idum = IA1 * (idum - k * IQ1) - k * IR1;
				 if (idum < 0) idum += IM1;
				 if (j < NTAB) iv[j] = idum;
			}
			iy = iv[0];
	 }
	 k = idum / IQ1;             //Start here when not initializing.
	 idum = IA1 * (idum - k * IQ1) - k * IR1; // Compute idum=(IA1*idum) % IM1 without overflows by Schrage's method.
	 if (idum < 0) idum += IM1;
	 k = idum2 / IQ2;
	 idum2 = IA2 * (idum2 - k * IQ2) - k * IR2; //Compute idum2=(IA2*idum) % IM2 likewise.
	 if (idum2 < 0) idum2 += IM2;
	 j = iy / NDIV;        //Will be in the range 0..NTAB-1.
	 iy= iv[j] - idum2;    //Here idum is shuffled, idum and idum2 are combined to generate output.
	 iv[j] = idum;
	 if (iy < 1) iy += IMM1;
	 if ((temp = AM * iy) > RNMX) return RNMX; //Because users don't expect endpoint values.
	 else return temp;
}

///////////////////////////////////////////////////////////////////////////////////////////
//
//      M Y   S T U F F
//
///////////////////////////////////////////////////////////////////////////////////////////

void Error(char *msg){cout << msg; exit(1);}
double mod2(double x, double y, double z){return (x*x)+(y*y)+(z*z);};                // Pythagoras
double mod(double x, double y, double z){return pow((x*x)+(y*y)+(z*z),0.5);};        // Pythagoras
double getGauss(){

	double gauss=-3.0;

	for(int i=0; i<6; ++i){
		gauss += ran2();
	}
	return gauss;
}

void setq(int *q)
{
	// get q[p] which is a shuffled list of the p's

	int i, trial;

	for(i=0;i<NPART;++i)
		q[i]=-1;

	i=0;
	while(i<NPART){
		trial = (int) (ran2() * NPART);
		if(q[trial]==-1){
			q[trial] = i;
			++i;
		}
	}
}

double setStep()

	// tweak the steplength to keep a good acceptance ratio

{
	double ratio;
	extern double step, sidex;
				extern long unsigned int accepted, rejected;

	ratio = (double) accepted / (double) (rejected+accepted);

	if(ratio > 0.5)	// we are accepting too many moves
		step = step * 1.1;

	if(ratio < 0.4)	// we are rejecting too many moves
		step = step * 0.9;

	if(step>sidex/10.0)
		step = sidex/10.0;

		return ratio;
}



int myRint(float x){      // homemade rint
							// since it s floating point based
							// it s not got a lot of sig figs

	float remainder;
	int intx;

	remainder = x - (float) ((int) x);
	intx = (float) ((int) x) + ((int) 2.0 * remainder);

	//x += 0.5;
	//intx = (int) x;


	return intx;
}

void makePolydisp(double *psigma, double *pmass){

	// make the sigma of the colloids polydisperse

	for(int i=0; i<NPART; ++i){
		psigma[i] = 1.0 + (POLYDISP * sqrt(2.0) * getGauss() );
		pmass[i]  = 0.5 * PI * (double) pow(psigma[i],3.0);
		//mass[i]  =  psigma[i];
		//cout << i << " psigma " << psigma[i] << " mass " <<mass[i] << endl;
	}

}

void scaleXYZ(double *px, double *py, double *pz, double scale){

	for(long int i=0; i<NPART; ++i){px[i] = px[i] * scale;  py[i] = py[i] * scale;  pz[i] = pz[i] * scale;}
}


void wrapAround(double *x, double *y, double *z)                                // impose periodic BCs
{

	*x = *x - (sidex * myRint(*x*invSidex));
	*y = *y - (sidey * myRint(*y*invSidey));
	*z = *z - (sidez * myRint(*z*invSidez));
}

double getSepPeriodic2(double *px, double *py, double *pz, int i, int j){                    // pythagoras between points i and j
																																													 // with periodic BCs
	double dx, dy, dz;

	dx = px[i] - px[j];
	dy = py[i] - py[j];
	dz = pz[i] - pz[j];
	wrapAround(&dx, &dy, &dz);
	return mod2(dx, dy, dz);
}

double getSepPeriodic(double *px, double *py, double *pz, int i, int j){                    // pythagoras between points i and j
																																													 // with periodic BCs
	double dx, dy, dz;

	dx = px[i] - px[j];
	dy = py[i] - py[j];
	dz = pz[i] - pz[j];
	wrapAround(&dx, &dy, &dz);
	return mod(dx, dy, dz);
}

void neighbours(double *px, double *py, double *pz, int *nNeighbour, int *neighbourList)     // N O T  T E S T E D
{                                                                                             //   routine called for each particle
	extern double step;

	for(int i=0;i<NPART*NEIGHBOURS;++i) neighbourList[i] = NPART;
	for(int i=0;i<NPART;++i) nNeighbour[i] = 0;

	for(int i=0;i<NPART;++i){
		for(int j=i+1;j<NPART;++j){
			if(getSepPeriodic2(px, py, pz, i, j)<((rCut+(5.0*step))*(rCut+(5.0*step)))){
				neighbourList[(i*NEIGHBOURS)+nNeighbour[i]] = j;
				++nNeighbour[i];
				neighbourList[(j*NEIGHBOURS)+nNeighbour[j]] = i;
				++nNeighbour[j];
				if(nNeighbour[i] > NEIGHBOURS || nNeighbour[j] >= NEIGHBOURS){
					cout << " cutOff = " << rCut+(5.0*step);
					cout << " particle " << i << " has " << nNeighbour[i] << " neighbours " << endl;
					Error(" \n Neighbours : - too many neighbours!\n ");
				}
			} // end if within range of neighbour lists
		}
	}
}

int overInit(double x, double y, double z, double *px, double *py, double *pz, int p, double *psigma){       // check particles aren t overlapping up to p

	double dx, dy, dz;
	double psigmaij;

	for(int i=0; i<p; ++i){
		dx = x - px[i];
		dy = y - py[i];
		dz = z - pz[i];
		psigmaij = 0.5 * (psigma[i] + psigma[p]);
		wrapAround(&dx, &dy, &dz);
		if(mod2(dx, dy, dz) < psigmaij*psigmaij){
			// cout << " overlapped " << diff.Mod2() << endl;
			return 1;
		}
	}
	return 0;
}

int overCheck(double *px, double *py, double *pz, double *psigma)      
	// check polydisperse particles aren t overlapping
{
	for(int i=0; i<NPART; ++i){
		for(int j=i+1; j<NPART; ++j){
			if(getSepPeriodic(px, py, pz, i,j) < 0.5*(psigma[i]+psigma[j])){
				cout << " i " << i << " j " << j << " sep " << pow(getSepPeriodic2(px, py, pz, i,j),0.5);
				cout << " sep2 " << getSepPeriodic2(px, py, pz, i,j) << endl;
				cout << " psigma[i] "  << psigma[i] << " psigma[j] "  << psigma[j] << " sum / 2 " << 0.5*(psigma[i]+psigma[j]) << endl;
				cout << " px[i] " << px[i] << " px[j] " << px[j] << endl;
				Error("\n particles overlapped ! \n");
				}
	}}
	return 0;
}

void initRandom(double *px, double *py, double *pz, double *psigma)
{
		/* randomly fill space with polydisperse particles */

	double x,y,z;
	int p=0;

	printf("randomly filling space\n");

	while(p<NPART){
		x=((ran2()-0.5)*sidex);
		y=((ran2()-0.5)*sidey);
		z=((ran2()-0.5)*sidez);

		if(overInit(x,y,z,px,py,pz,p, psigma)==0){	// the new particle doesn t overlap: take trial values
			px[p]=x;  py[p]=y;  pz[p]=z;
			++p;
		}
	}
}

double getEnergy(double x, double y, double z, double *px, double *py, double *pz, int p, int *nNeighbour, int *neighbourList, double *psigma, double thisSigma)

				// sum potential energy for each particle
				// neighbour list version
{
	double dx, dy, dz;
	double energy = 0.0;
	double r, r2;
	double psigmaij;



	for(int i=0; i<nNeighbour[p]; ++i){
			psigmaij = 0.5 * (thisSigma + psigma[neighbourList[i+(p*NEIGHBOURS)]]);
			dx = x - px[neighbourList[i+(p*NEIGHBOURS)]];
			dy = y - py[neighbourList[i+(p*NEIGHBOURS)]];
			dz = z - pz[neighbourList[i+(p*NEIGHBOURS)]];
			wrapAround(&dx, &dy, &dz);
			r2 = mod2(dx, dy, dz);
			if(r2<psigmaij*psigmaij) {
				energy += 1e10;
				}
			else{
				r = sqrt(r2);
				if(r<psigmaij+STEPRANGE) energy+=STEPHEIGHT;
				else if(r-psigmaij+1.0 < 1.0){
					cout << " r " << r << " psigmaij " << psigmaij << endl;
					Error("too close"); }
				}
			//cout << " p " << p << " i " << i << " r " << r << " psigmaij " << psigmaij << " psigmap ";
			//cout << psigma[p] << " psigmai " << psigma[i] << endl;
	}
	return energy;              // double coz haven t double-counted
}


/*double getEnergyAll(double *px, double *py, double *pz, double *psigma)            // sum potential energy for each particle
{                                                           // polydisperse
				extern double radius, rCut, rCutYuk;
				double energy = 0.0;
				double psigmaij, r, r2;

				for(int i=0; i<NPART; ++i){
					for(int j=i+1; j<NPART; ++j){
						psigmaij = 0.5 * (psigma[j] + psigma[i]);
						r2 = getSepPeriodic2(px, py, pz, i,j);
						if(r2<1.01*psigmaij*psigmaij) {           // L-J NO HARD CORE
							energy += 1e8;
							}
						else{
							r = sqrt(r2);
						}
				}}

				return energy;              // double coz haven t double-counted
}   */

double getEnergyAllFat(double *px, double *py, double *pz, double *psigma, double rScale)            // sum potential energy for each particle
{                                                           // polydisperse
	extern double radius, rCut, rCutYuk;
	double energy = 0.0;
	double psigmaij, r, r2;
	double invRscale2 = 1.0/(rScale*rScale);

	for(int i=0; i<NPART; ++i){
		for(int j=i+1; j<NPART; ++j){
			psigmaij = 0.5 * (psigma[j] + psigma[i]);
			r2 = getSepPeriodic2(px, py, pz, i,j);
			if(r2<1.0*psigmaij*psigmaij*invRscale2) {
				energy += 1e8;
				}
	}}

	return energy;              // double coz haven t double-counted
}


void advance(double *px, double *py, double *pz, double *newX, double *newY, double *newZ, int p)
	// simple mc routine to advance particles
{
	// go through list...this IS EQUIVALENT to
	//	randomly selecting particles
	extern double step;
	double dx, dy, dz;

	if(ran2() >= 0.5) dx = step; else dx = -step;
	if(ran2() >= 0.5) dy = step; else dy = -step;
	if(ran2() >= 0.5) dz = step; else dz = -step;

	 *newX = px[p] + dx;              // trial new location
	 *newY = py[p] + dy;
	 *newZ = pz[p] + dz;

	 wrapAround(newX, newY, newZ);
}

void accept(double *px, double *py, double *pz, double trialX, double trialY, double trialZ, int p, int *nNeighbour, int *neighbourList, double *psigma)
{
	// choose whether to accept mc move
	//always accept if newenergy < energy (exp > 1)
	//randomly accept else...metropolis...

	double newEnergy, energy, ran;

	energy    = getEnergy(px[p],  py[p],  pz[p],  px, py, pz, p, nNeighbour, neighbourList, psigma, psigma[p]);
	newEnergy = getEnergy(trialX, trialY, trialZ, px, py, pz, p, nNeighbour, neighbourList, psigma, psigma[p]);

	if(ran2() < exp(energy-newEnergy)){
		px[p] = trialX;
		py[p] = trialY;
		pz[p] = trialZ;
		++accepted;
	}
	else
		++rejected;
}




void readxyz(char *filenamepath, double *px, double *py, double *pz)  // output single float valued arrays
{                                                // in gopenmol xyz format
	ifstream File(filenamepath);  // Creates an ofstream object named myFile
	int nPart;
	char c;
	double x, y, z;

	if (!File){ // Always test file open
		Error("readxyz : Error opening file");
	}

	File >> nPart;
	cout << " \n reading in  " << nPart << " particles " << endl;

	for(long int i=0; i< NPART; ++i){
		File >> c >> x >> y >> z;
		px[i] = x; 	py[i] = y;	  pz[i] = z;
	}

	File.close();
}


void read(char *filenamepath, double *psigma, int index)  // output single float valued arrays
{                                                // in gopenmol xyz format
	ifstream File(filenamepath);  // Creates an ofstream object named myFile
	int n;
	char c;

	if (!File)  Error("read : Error opening file");

	for(long int i=0; i< index; ++i)
		File >> n >>  psigma[i];

	File.close();
}
void writexyz(char *filenamepath, double *x, double *y, double *z)  // output single float valued arrays
{                                                // in gopenmol xyz format
	ofstream File(filenamepath);  // Creates an ofstream object named myFile

	//cout << " \n Writexyz : trying to write to " <<  filenamepath << endl;

	if (!File) // Always test file open
			Error("Writexyz : Error opening file");
	// Error(" \n OK! \n ");
	File << NPART << endl;
	//File << " t " << 0 << " sidex "	<< sidex <<	" sidey " << sidey << " sidez "	<< sidez <<	" volume " << sidex * sidey * sidez <<	" rho " << (double) NPART / (sidex * sidey * sidez)  << endl;
	File << " initial config d0 rho " <<  rho << " T " << 0 << " P " << 0 << " themin2 1.0404 " << endl;

	for(long int i=0; i< NPART; ++i)
		File << "A" << "\t" << x[i] << "\t" << y[i] << "\t" << z[i]  << endl;

	File.close();
}

void write(char *filenamepath, double *a, double scale, int index)  // output single double valued arrays
{
	ofstream File(filenamepath);  // Creates an ofstream object named myFile

	if (!File) // Always test file open
			Error("Write : Error opening file");

			cout << " \n write : printing to " <<  filenamepath << endl;

	for(int i=0; i<index; ++i) {
		File << (double) i * scale << "\t" <<a[i] <<endl;
	}

	File.close();
}

void write(char *filenamepath, double *psigma, int index)  // output single float valued arrays
{                                                // in gopenmol xyz format
	ofstream File(filenamepath);  // Creates an ofstream object named myFile
	int n;
	char c;

	if (!File)  Error("read : Error opening file");

	for(long int i=0; i< index; ++i) File << i << "\t" << psigma[i] << endl;

	File.close();
}

///////////////////////////////////////////////////////////////////////////////////////////
//
//      M A I N
//
///////////////////////////////////////////////////////////////////////////////////////////

int main()
{
				double r, rScale;                                                       // scaling factor
				double realTime=0.0, tBrown;                                            // accepted time, brownian time
				long t=0;
				double ratio;
				int q[NPART];                                                           // particle order
				double px[NPART], py[NPART], pz[NPART];                                 // particle locations
				double px0[NPART], py0[NPART], pz0[NPART];                              // particle locations at start of production run
				double psigma[NPART], pmass[NPART];                                     // particle diameter and mass - mass not yet used
				double trialX, trialY, trialZ;
				double energy;
				double eta=ETA, etaNew;                                                 // current and new eta
				int *neighbourList;                                                     // neighbour list
				int nNeighbour[NPART];                                                  // # neighbours
				long seed;
				char input[1000], output[1000], outputXmol[1000], outputDat[1000];

				//randomize();                                                            // seed sran2 off the clock
				seed    = (long) (rand() * 10000);
				sran2(seed);

				cout    << " \n Paddy Royall Feb 2008 \n HS one component polydisp crushing \n ";
				cout    << " \n FATTING \n ";

				neighbourList = new int[NPART*NEIGHBOURS];
				sidez = sidex = sidey = pow((double)NPART/rho, 0.333333);
				invSidez = invSidex = invSidey = 1.0/sidex;
				etaNew = eta+ETADIFF;
				printf("\n NPART %d ETA %.2f sidex %.3f  ", NPART, ETA, sidex);

				if(ETA>0.31) {
					 sprintf(input, "coord_e%.4f_n%d_poly%.2f.xyz"  , ETA, NPART, POLYDISP);
					 cout << " dense - reading in " << input << endl;
					 readxyz(input, px, py, pz);
					 sprintf(input, "sigma_e%.2f_n%d_poly%.2f.dat"  , ETA, NPART, POLYDISP);
					 cout << " and for sigma - reading in " << input << endl;
					 read(input, psigma, NPART);
				}

				else{
						makePolydisp(psigma, pmass);
						initRandom(px, py, pz, psigma);
				}



				while(eta<ETAFINAL){
					
					neighbours(px, py, pz, nNeighbour, neighbourList);
					rejected = accepted = 0;
					setq(q);
					for(int i=0; i<NPART; ++i){
						advance(px, py, pz, &trialX, &trialY, &trialZ, q[i]);
						accept(px, py, pz, trialX, trialY, trialZ, q[i], nNeighbour, neighbourList, psigma);
						//accept(px, py, pz, px[q[i]], py[q[i]], pz[q[i]], q[i], nNeighbour, neighbourList, psigma);
					}
					ratio = (double) accepted / (double) (rejected+accepted);
					setStep();
					rScale = pow(eta/etaNew, 0.333333333);
					//cout << " rScale " << rScale << endl;
					energy = getEnergyAllFat(px, py, pz, psigma, rScale);

					if(energy<STEPHEIGHT && eta<ETAFINAL){
						cout << " energy " << energy << " OK - lets squish - should be no overlaps " << endl;
						sidex     = sidey    = sidez    = (sidex * rScale);
						invSidex  = invSidey = invSidez = (invSidex / rScale);
						scaleXYZ(px, py, pz, rScale);
						eta    = etaNew;
						etaNew = eta+ETADIFF;
						cout << " now eta " << eta <<  " rScale " << rScale  << endl;
					}

					overCheck(px, py, pz, psigma);
					++t;
				}                                                           // end equilibration loop
				sprintf(output, "coord_e%.5f_n%d_poly%.2f.xyz"  , ETAFINAL, NPART, POLYDISP);  writexyz(output, px, py, pz);
				sprintf(output, "sigma_e%.5f_n%d_poly%.2f.dat"  , ETAFINAL, NPART, POLYDISP);  write(output, psigma, NPART);

				sprintf(output, "d0_coord_input.xyz");  writexyz(output, px, py, pz);
				sprintf(output, "d0_sigma_input.dat");  write(output, psigma, NPART);


				cout << "\n\n FIN \n\n";

				return 0;

}
