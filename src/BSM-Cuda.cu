#include "legendre_rule.h"
#include "matrixelements.h"
#include <stdio.h>
#include <vector>

__device__ double me;
__device__ double m2;
__device__ double Gf;
__device__ double g;
__device__ int orderE1;
__device__ int orderE2;
__device__ int orderct1;
__device__ int orderct2;
__device__ int orderct3;
__device__ int orderp2;
__device__ int orderp3;
__device__ double m;

struct Settings {
  double me = 0.000511;
  double m2 = 0.01;
  double Gf = 0.0000116637;
  double g = 1.0;
  int orderE1 = 32;
  int orderE2 = 32;
  int orderct1 = 32;
  int orderct2 = 32;
  int orderct3 = 32;
  int orderp2 = 32;
  int orderp3 = 32;
  double m = 0.105660;
  int done = 0;
  string filename = "	";
  bool operator==(const Settings &a) const {
    return (m2 == a.m2 && done == a.done && orderE1 == a.orderE1 &&
            orderE2 == a.orderE2 && orderct1 == a.orderct1 &&
            orderct2 == a.orderct2 && orderct3 == a.orderct3 &&
            orderp2 == a.orderp2 && orderp3 == a.orderp3);
  }
} sett;

// Calculate derivative appearing in the phasespace integral
__device__ double Deriv(double E1, double E2, double ct1, double ct2,
                        double ct3, double ph2, double ph3) {
  return 2 * (E1 + E2 - m -
              (ct2 * ct3 + cos(ph2 - ph3) * pow(1 - pow(ct2, 2), 0.5) *
                               pow(1 - pow(ct3, 2), 0.5)) *
                  pow(pow(E2, 2) - pow(m2, 2), 0.5) -
              pow(pow(E1, 2) - pow(me, 2), 0.5) *
                  (ct1 * ct3 + pow(1 - pow(ct1, 2), 0.5) *
                                   pow(1 - pow(ct3, 2), 0.5) * sin(ph3)));
}

// Solve kinetic equation for E3
__device__ double CalcE3(double E1, double E2, double ct1, double ct2,
                         double ct3, double ph2, double ph3) {
  return -(pow(E1 + E2 - m -
                   (ct2 * ct3 + cos(ph2 - ph3) * pow(1 - pow(ct2, 2), 0.5) *
                                    pow(1 - pow(ct3, 2), 0.5)) *
                       pow(pow(E2, 2) - pow(m2, 2), 0.5) -
                   pow(pow(E1, 2) - pow(me, 2), 0.5) *
                       (ct1 * ct3 + pow(1 - pow(ct1, 2), 0.5) *
                                        pow(1 - pow(ct3, 2), 0.5) * sin(ph3)),
               -1) *
           (2 * E1 * E2 - 2 * E1 * m - 2 * E2 * m + pow(m, 2) + pow(m2, 2) +
            pow(me, 2) -
            2 * pow(pow(E2, 2) - pow(m2, 2), 0.5) *
                pow(pow(E1, 2) - pow(me, 2), 0.5) *
                (ct1 * ct2 + pow(1 - pow(ct1, 2), 0.5) *
                                 pow(1 - pow(ct2, 2), 0.5) * sin(ph2)))) /
         2.;
}

// Implements a test for physicality, thus avoiding complicated region
__device__ int isPhysical(double E1, double E2, double E3) {
  if (E1 < me || E2 < 0 || E3 < 0 || E1 + E2 + E3 > m)
    return 0;
  return 1;
}

// Calculate the integrals on the GPU
__global__ void CalcRes(double *d_wE1, double *d_xE1, double *d_wE2,
                        double *d_xE2, double *d_wct1, double *d_xct1,
                        double *d_wct2, double *d_xct2, double *d_wct3,
                        double *d_xct3, double *d_wp2, double *d_xp2,
                        double *d_wp3, double *d_xp3, double *d_res) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double weight = 0.0;
  d_res[i] = 0.0;
  double E1, E2, E3, ct1, ct2, ct3, ph2, ph3;
  if (i < orderE1 * orderE2 * orderct1) {
    int iE1 = i % orderE1;
    int iCt1 = (i % (orderct1 * orderE1)) / orderE1;
    int iE2 = i / (orderE1 * orderct1);
    for (int iCt2 = 0; iCt2 < orderct2; iCt2++) {
      for (int iCt3 = 0; iCt3 < orderct3; iCt3++) {
        for (int ip2 = 0; ip2 < orderp2; ip2++) {
          for (int ip3 = 0; ip3 < orderp3; ip3++) {
            E1 = d_xE1[iE1];
            E2 = d_xE2[iE2];
            ct1 = d_xct1[iCt1];
            ct2 = d_xct2[iCt2];
            ct3 = d_xct3[iCt3];
            ph2 = d_xp2[ip2];
            ph3 = d_xp3[ip3];
            E3 = CalcE3(E1, E2, ct1, ct2, ct3, ph2, ph3);
            if (isPhysical(E1, E2, E3)) {
              weight = pow(2.0 * M_PI, -7) / 8.0 * sqrt(E1 * E1 - me * me) *
                       sqrt(E2 * E2 - m2 * m2) * E3 /
                       abs(Deriv(E1, E2, ct1, ct2, ct3, ph2, ph3)) *
                       d_wct2[iCt2] * d_wct3[iCt3] * d_wp2[ip2] * d_wp3[ip3];
              d_res[i] += weight * MatrixElements::M2ScalarToLepton(
                                       E1, E2, E3, ct1, ct2, ct3, ph2, ph3);
            }
          }
        }
      }
    }
  }
}

// Carry out all integrals on the GPU, pull back results
void IntegrateOnGPU(const char *Filename) {
  // Allocate memory for the quadrature nodes and results
  double *d_wE1, *d_xE1, *d_wE2, *d_xE2, *d_wct1, *d_xct1, *d_wct2, *d_xct2,
      *d_wct3, *d_xct3, *d_wp2, *d_xp2, *d_wp3, *d_xp3, *d_res;
  double *wE1 = (double *)malloc(sett.orderE1 * sizeof(double));
  double *xE1 = (double *)malloc(sett.orderE1 * sizeof(double));
  double *wE2 = (double *)malloc(sett.orderE2 * sizeof(double));
  double *xE2 = (double *)malloc(sett.orderE2 * sizeof(double));
  double *wct1 = (double *)malloc(sett.orderct1 * sizeof(double));
  double *xct1 = (double *)malloc(sett.orderct1 * sizeof(double));
  double *wct2 = (double *)malloc(sett.orderct2 * sizeof(double));
  double *xct2 = (double *)malloc(sett.orderct2 * sizeof(double));
  double *wct3 = (double *)malloc(sett.orderct3 * sizeof(double));
  double *xct3 = (double *)malloc(sett.orderct3 * sizeof(double));
  double *wp2 = (double *)malloc(sett.orderp2 * sizeof(double));
  double *xp2 = (double *)malloc(sett.orderp2 * sizeof(double));
  double *wp3 = (double *)malloc(sett.orderp3 * sizeof(double));
  double *xp3 = (double *)malloc(sett.orderp3 * sizeof(double));
  double *resVec = (double *)malloc(sett.orderE1 * sett.orderE2 *
                                    sett.orderct1 * sizeof(double));

  // Allocate memory on the GPU for the nodes and results
  cudaMalloc(&d_wE1, sett.orderE1 * sizeof(double));
  cudaMalloc(&d_xE1, sett.orderE1 * sizeof(double));
  cudaMalloc(&d_wE2, sett.orderE2 * sizeof(double));
  cudaMalloc(&d_xE2, sett.orderE2 * sizeof(double));
  cudaMalloc(&d_wct1, sett.orderct1 * sizeof(double));
  cudaMalloc(&d_xct1, sett.orderct1 * sizeof(double));
  cudaMalloc(&d_wct2, sett.orderct2 * sizeof(double));
  cudaMalloc(&d_xct2, sett.orderct2 * sizeof(double));
  cudaMalloc(&d_xct3, sett.orderct3 * sizeof(double));
  cudaMalloc(&d_wct3, sett.orderct3 * sizeof(double));
  cudaMalloc(&d_wp2, sett.orderp2 * sizeof(double));
  cudaMalloc(&d_xp2, sett.orderp2 * sizeof(double));
  cudaMalloc(&d_wp3, sett.orderp3 * sizeof(double));
  cudaMalloc(&d_xp3, sett.orderp3 * sizeof(double));
  cudaMalloc(&d_res,
             sett.orderE1 * sett.orderE2 * sett.orderct1 * sizeof(double));

  // Calculate quadrature nodes
  cgqf(sett.orderct1, 1, 0, 0, -1.0, 1.0, xct1, wct1);
  cgqf(sett.orderct2, 1, 0, 0, -1.0, 1.0, xct2, wct2);
  cgqf(sett.orderct3, 1, 0, 0, -1.0, 1.0, xct3, wct3);
  cgqf(sett.orderp2, 1, 0, 0, 0.0, 2.0 * M_PI, xp2, wp2);
  cgqf(sett.orderp3, 1, 0, 0, 0.0, 2.0 * M_PI, xp3, wp3);
  cgqf(sett.orderE1, 1, 0, 0, sett.me, sett.m / 2, xE1, wE1);
  cgqf(sett.orderE2, 1, 0, 0, sett.m2, (sett.m + sett.m2) / 2.0, xE2, wE2);

  // Transfere to GPU
  cudaMemcpy(d_wE1, wE1, sett.orderE1 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xE1, xE1, sett.orderE1 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wE2, wE2, sett.orderE2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xE2, xE2, sett.orderE2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wct1, wct1, sett.orderct1 * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_xct1, xct1, sett.orderct1 * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_wct2, wct2, sett.orderct2 * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_xct2, xct2, sett.orderct2 * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_wct3, wct3, sett.orderct3 * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_xct3, xct3, sett.orderct3 * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_wp2, wp2, sett.orderp2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xp2, xp2, sett.orderp2 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wp3, wp3, sett.orderp3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xp3, xp3, sett.orderp3 * sizeof(double), cudaMemcpyHostToDevice);

  // Break work into chunks and solve each on the GPU
  int N = sett.orderE1 * sett.orderE2 * sett.orderct1;
  CalcRes<<<(N + 511) / 512, 512>>>(d_wE1, d_xE1, d_wE2, d_xE2, d_wct1, d_xct1,
                                    d_wct2, d_xct2, d_wct3, d_xct3, d_wp2,
                                    d_xp2, d_wp3, d_xp3, d_res);

  // Pull back results from the GPU
  cudaMemcpy(resVec, d_res, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Calculate the total decay width to crosscheck with montecarlo estimates
  double res = 0.0;
  for (int iE1 = 0; iE1 < sett.orderE1; iE1++) {
    for (int iCt1 = 0; iCt1 < sett.orderct1; iCt1++) {
      for (int iE2 = 0; iE2 < sett.orderE2; iE2++) {
        res += resVec[iE1 + iE2 * sett.orderE1 * sett.orderct1 +
                      iCt1 * sett.orderE1] *
               wE1[iE1] * wE2[iE2] * wct1[iCt1] / (2.0 * sett.m);
      }
    }
  }
  // Write the complete spectrum to a file
  FILE *pFile;
  pFile = fopen(Filename, "w");
  fprintf(pFile,
          "######	%.12E	%f	%f	%i	%i	%i	"
          "%i	%i	%i	%i\n",
          res, sett.m2, sett.m, sett.orderE1, sett.orderct1, sett.orderE2,
          sett.orderct2, sett.orderct3, sett.orderp2, sett.orderp3);
  for (int iE1 = 0; iE1 < sett.orderE1; iE1++) {
    for (int iCt1 = 0; iCt1 < sett.orderct1; iCt1++) {
      double buff = 0.0;
      for (int iE2 = 0; iE2 < sett.orderE2; iE2++) {
        buff += resVec[iE1 + iE2 * sett.orderE1 * sett.orderct1 +
                       iCt1 * sett.orderE1] *
                wE1[iE1] * wE2[iE2] * wct1[iCt1];
      }
      fprintf(pFile, "%.12E,%.12E,%.12E\n", xE1[iE1] * 2.0 / sett.m, xct1[iCt1],
              buff);
    }
  }
  fclose(pFile);
  printf("Total Width : %E\n Created File : %s", res, Filename);

  // Free allocated memory
  free(wE1);
  free(xE1);
  free(wE2);
  free(xE2);
  free(wct1);
  free(xct1);
  free(wct2);
  free(xct2);
  free(wct3);
  free(xct3);
  free(wp2);
  free(xp2);
  free(wp3);
  free(xp3);

  cudaFree(d_wE1);
  cudaFree(d_xE1);
  cudaFree(d_wE2);
  cudaFree(d_xE2);
  cudaFree(d_wct1);
  cudaFree(d_xct1);
  cudaFree(d_wct2);
  cudaFree(d_xct2);
  cudaFree(d_wct3);
  cudaFree(d_xct3);
  cudaFree(d_wp2);
  cudaFree(d_xp2);
  cudaFree(d_wp3);
  cudaFree(d_xp3);
}

// Estimate the runtime of the given choice of number of nodes
void EstimateRuntime() {
  double i = (5580000.0 * sett.orderE1 * sett.orderE2 * sett.orderct1 *
              sett.orderct2 * sett.orderct3 * sett.orderp2 * sett.orderp3) /
             34359738368;
  printf("%ih%im%is%ims\n", (int)(i / (60 * 60 * 1000)),
         (int)(i / (1000 * 60)) % 60, (int)(i / 1000) % 60, ((int)i) % 1000);
}

// Initialise fixed parameters on the device
void InitParameters() {
  cudaMemcpyToSymbol(m, &sett.m, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(me, &sett.me, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(m2, &sett.m2, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Gf, &sett.Gf, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(g, &sett.g, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderE1, &sett.orderE1, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderE2, &sett.orderE2, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderct1, &sett.orderct1, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderct2, &sett.orderct2, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderct3, &sett.orderct3, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderp2, &sett.orderp2, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(orderp3, &sett.orderp3, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
}

// Read space-seperated field of a line
const char *getfield(char *line, int num) {
  const char *tok;
  for (tok = strtok(strdup(line), "	"); tok && *tok;
       tok = strtok(NULL, "	\n")) {
    if (!--num)
      return tok;
  }
  return NULL;
}

// Parse char* line to settings se
void ReadSettings(char *line, struct Settings &se) {
  se.m2 = atof(getfield(line, 1));
  se.orderE1 = atoi(getfield(line, 2));
  se.orderE2 = atoi(getfield(line, 3));
  se.orderct1 = atoi(getfield(line, 4));
  se.orderct2 = atoi(getfield(line, 5));
  se.orderct3 = atoi(getfield(line, 6));
  se.orderp2 = atoi(getfield(line, 7));
  se.orderp3 = atoi(getfield(line, 8));
  se.done = atoi(getfield(line, 9));
  if (se.done)
    se.filename = string(getfield(line, 10));
}

// Read schedule from external file to avoid recompiling
void ReadSchedule(vector<Settings> &listSett) {
  listSett.clear();
  FILE *stream = fopen("schedule.txt", "r");
  char line[1024];
  while (fgets(line, 1024, stream)) {

    char *tmp = strdup(line);
    struct Settings se;
    ReadSettings(tmp, se);
    listSett.push_back(se);
    free(tmp);
  }
  fclose(stream);
}

// Write new schedule-status to file
void WriteSchedule(vector<Settings> listSett) {
  FILE *stream = fopen("schedule.txt", "w");
  for (int i = 0; i < static_cast<int>(listSett.size()); i++) {
    Settings curr = listSett[i];
    if (curr.done)
      fprintf(stream,
              "%f	%i	%i	%i	%i	%i	%i	"
              "%i	%i	%s\n",
              curr.m2, curr.orderE1, curr.orderE2, curr.orderct1, curr.orderct2,
              curr.orderct3, curr.orderp2, curr.orderp3, curr.done,
              curr.filename.c_str());
    else
      fprintf(
          stream,
          "%f	%i	%i	%i	%i	%i	%i	%i	%i\n",
          curr.m2, curr.orderE1, curr.orderE2, curr.orderct1, curr.orderct2,
          curr.orderct3, curr.orderp2, curr.orderp3, curr.done);
  }
  fclose(stream);
}

// Check for new work and load settings
int FetchNewWork(vector<Settings> &listSett) {
  ReadSchedule(listSett);
  for (int i = 0; i < static_cast<int>(listSett.size()); i++) {
    if (!listSett[i].done) {
      sett = listSett[i];
      printf("\n Found new Work\n");
      return 1;
    }
  }
  return 0;
}

// Runs integrations as long as new work is scheduled
void RunScheduledJobs() {
  srand(time(NULL));
  vector<Settings> listSett;
  while (FetchNewWork(listSett)) {
    InitParameters();
    EstimateRuntime();
    string filenam = "Results/Scalar/" + to_string(rand() % 100000) + ".csv";
    IntegrateOnGPU(filenam.c_str());
    ReadSchedule(listSett);
    for (int i = 0; i < static_cast<int>(listSett.size()); i++) {

      if (sett == listSett[i]) {
        listSett[i].done = 1;
        listSett[i].filename = filenam;
        break;
      }
    }
    WriteSchedule(listSett);
  }
}

// Serves as a quick test for debug-purposes
void Quicktest() {
  sett.orderE1 = 10;
  sett.orderE2 = 10;
  sett.orderct1 = 10;
  sett.orderct2 = 10;
  sett.orderct3 = 10;
  sett.orderp2 = 10;
  sett.orderp3 = 10;
  sett.m2 = 0.01;
  EstimateRuntime();
  InitParameters();
  IntegrateOnGPU("/tmp/test.txt");
}

int main(int argc, char *argv[]) {

  Quicktest();
  return 0;
}
