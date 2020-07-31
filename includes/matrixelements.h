#pragma once

using namespace std;

extern __device__ double me;
extern __device__ double m2;
extern __device__ double Gf;
extern __device__ double g;
extern __device__ int orderE1;
extern __device__ int orderE2;
extern __device__ int orderct1;
extern __device__ int orderct2;
extern __device__ int orderct3;
extern __device__ int orderp2;
extern __device__ int orderp3;
extern __device__ double m;

class MatrixElements {
public:
  static __device__ double M2ScalarToLepton(double E1, double E2, double E3,
                                            double ct1, double ct2, double ct3,
                                            double ph2, double ph3);
  static __device__ double
  M2VectorTauToElectronTVCoupling(double E1, double E2, double E3, double ct1,
                                  double ct2, double ct3, double ph2,
                                  double ph3);
  static __device__ double M2VectorElectronRLCoupling(double E1, double E2,
                                                      double E3, double ct1,
                                                      double ct2, double ct3,
                                                      double ph2, double ph3);
  static __device__ double M2VectorMuonLCoupling(double E1, double E2,
                                                 double E3, double ct1,
                                                 double ct2, double ct3,
                                                 double ph2, double ph3);
  static __device__ double M2VectorElectronLCoupling(double E1, double E2,
                                                     double E3, double ct1,
                                                     double ct2, double ct3,
                                                     double ph2, double ph3);
  static __device__ double M2VectorMuonRCoupling(double E1, double E2,
                                                 double E3, double ct1,
                                                 double ct2, double ct3,
                                                 double ph2, double ph3);
  static __device__ double M2VectorElectronRCoupling(double E1, double E2,
                                                     double E3, double ct1,
                                                     double ct2, double ct3,
                                                     double ph2, double ph3);
  static __device__ double M2ScalarMuonCoupling(double E1, double E2, double E3,
                                                double ct1, double ct2,
                                                double ct3, double ph2,
                                                double ph3);
  static __device__ double M2ScalarElectronCoupling2(double E1, double E2,
                                                     double E3, double ct1,
                                                     double ct2, double ct3,
                                                     double ph2, double ph3);
  static __device__ double M2ScalarElectronCoupling(double E1, double E2,
                                                    double E3, double ct1,
                                                    double ct2, double ct3,
                                                    double ph2, double ph3);
  static __device__ double M2VectorMuonRLCoupling(double E1, double E2,
                                                  double E3, double ct1,
                                                  double ct2, double ct3,
                                                  double ph2, double ph3);
};
