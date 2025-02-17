#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>

//
// Device-side helpers
//
KOKKOS_INLINE_FUNCTION
void identityMatrix(double I[3][3]) {
  I[0][0] = 1.0; I[0][1] = 0.0; I[0][2] = 0.0;
  I[1][0] = 0.0; I[1][1] = 1.0; I[1][2] = 0.0;
  I[2][0] = 0.0; I[2][1] = 0.0; I[2][2] = 1.0;
}

KOKKOS_INLINE_FUNCTION
void outerProduct(const double a[3], const double b[3], double result[3][3]) {
  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
      result[i][j] = a[i] * b[j];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void epsilonMatrix(const double r[3], double e[3][3]) {
  e[0][0] =  0.0;    e[0][1] = -r[2];   e[0][2] =  r[1];
  e[1][0] =  r[2];   e[1][1] =  0.0;    e[1][2] = -r[0];
  e[2][0] = -r[1];   e[2][1] =  r[0];   e[2][2] =  0.0;
}

KOKKOS_INLINE_FUNCTION
void addMatrices(const double a[3][3], const double b[3][3], double result[3][3]) {
  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void scaleMatrix(const double m[3][3], double scalar, double result[3][3]) {
  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
      result[i][j] = m[i][j] * scalar;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void matVec3(const double M[3][3], const double v[3], double out[3]) {
  for(int i=0; i<3; ++i) {
    out[i] = M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2];
  }
}

//
// This is the device-side version of your computeRPYFarFieldVelocity.
// Return in "velocity[6]" = (u_x, u_y, u_z, w_x, w_y, w_z).
//
KOKKOS_INLINE_FUNCTION
void computeRPYFarFieldVelocity(const double center1[3],
                                const double center2[3],
                                const double force2[3],
                                const double torque2[3],
                                double velocity[6])
{
  const double a   = 1.0; // radius
  const double pi  = M_PI;

  // dr = center1 - center2
  double dr[3];
  dr[0] = center1[0] - center2[0];
  dr[1] = center1[1] - center2[1];
  dr[2] = center1[2] - center2[2];

  double r_sq = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
  double r    = sqrt(r_sq);

  // If not in the far field
  if(r <= 2.0*a) {
    // Return all zero
    for(int i=0; i<6; ++i) {
      velocity[i] = 0.0;
    }
    return;
  }

  // r_hat = dr / r
  double r_hat[3];
  r_hat[0] = dr[0]/r; 
  r_hat[1] = dr[1]/r; 
  r_hat[2] = dr[2]/r;

  double r3 = r_sq * r;    // r^3
  double r5 = r3 * r_sq;   // r^5 (just in case we need it)

  //
  // Build sub-blocks
  //
  double I[3][3]; 
  identityMatrix(I);

  // muTT
  double TT_identity = (1.0/(8.0*pi*r)) * (1.0 + (2.0*a*a)/(3.0*r_sq));
  double TT_rhat     = (1.0/(8.0*pi*r)) * (1.0 - (2.0*a*a)/r_sq);

  double I_scaled[3][3], rhat_rhat[3][3], rhat_rhat_scaled[3][3];
  scaleMatrix(I, TT_identity, I_scaled);
  outerProduct(r_hat, r_hat, rhat_rhat);
  scaleMatrix(rhat_rhat, TT_rhat, rhat_rhat_scaled);

  double muTT[3][3];
  addMatrices(I_scaled, rhat_rhat_scaled, muTT);

  // muRR
  double RR_identity = -1.0/(16.0*pi*r3);
  double RR_rhat     =  3.0/(16.0*pi*r3);
  scaleMatrix(I, RR_identity, I_scaled);
  scaleMatrix(rhat_rhat, RR_rhat, rhat_rhat_scaled);

  double muRR[3][3];
  addMatrices(I_scaled, rhat_rhat_scaled, muRR);

  // muRT
  double muRT_scale = 1.0/(8.0*pi*r_sq);
  double E[3][3], E_scaled[3][3];
  epsilonMatrix(r_hat, E);
  scaleMatrix(E, muRT_scale, E_scaled);

  // muRT^T = transpose(E_scaled)
  double muRTT[3][3];
  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
      muRTT[i][j] = E_scaled[j][i];
    }
  }

  // Multiply to get velocity
  double u_part1_F[3], u_part1_T[3];
  matVec3(muTT, force2,     u_part1_F);
  matVec3(muRTT, torque2,   u_part1_T);

  double w_part1_F[3], w_part1_T[3];
  matVec3(E_scaled, force2, w_part1_F);
  matVec3(muRR, torque2,    w_part1_T);

  double u[3], w[3];
  for(int i=0; i<3; ++i) {
    u[i] = u_part1_F[i] + u_part1_T[i];
    w[i] = w_part1_F[i] + w_part1_T[i];
  }

  // velocity[0..2] = u, velocity[3..5] = w
  velocity[0] = u[0];
  velocity[1] = u[1];
  velocity[2] = u[2];
  velocity[3] = w[0];
  velocity[4] = w[1];
  velocity[5] = w[2];
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;

    // We'll do N calls in parallel. We'll measure it multiple times.
    const int N = 2'000'000; 

    // Make device arrays
    Kokkos::View<double*[3], MemSpace> center1("center1", N);
    Kokkos::View<double*[3], MemSpace> center2("center2", N);
    Kokkos::View<double*[3], MemSpace> force2("force2",   N);
    Kokkos::View<double*[3], MemSpace> torque2("torque2", N);
    Kokkos::View<double*[6], MemSpace> velocity("velocity", N);

    // Small warmup (not timed)
    {
      const int warmupIters = 5;
      for(int iter=0; iter<warmupIters; ++iter) {
        // Re-init data
        Kokkos::parallel_for("WarmupInit", Kokkos::RangePolicy<>(0,N),
          KOKKOS_LAMBDA(const int i) {
            center1(i,0) = 0.0; 
            center1(i,1) = 0.0; 
            center1(i,2) = 0.0;
            center2(i,0) = 2.0; 
            center2(i,1) = 1.0; 
            center2(i,2) = 3.0;
            force2(i,0)  = 1.0; 
            force2(i,1)  = 2.0; 
            force2(i,2)  = -6.0;
            torque2(i,0) = 0.0; 
            torque2(i,1) = 0.0; 
            torque2(i,2) = 1.0;
          }
        );
        Kokkos::fence();

        // Warmup kernel
        Kokkos::parallel_for("WarmupKernel", Kokkos::RangePolicy<>(0,N),
          KOKKOS_LAMBDA(const int i){
            double c1[3], c2[3], f2[3], t2[3], out[6];
            for(int j=0; j<3; ++j){
              c1[j] = center1(i,j);
              c2[j] = center2(i,j);
              f2[j] = force2(i,j);
              t2[j] = torque2(i,j);
            }
            computeRPYFarFieldVelocity(c1, c2, f2, t2, out);
            for(int j=0; j<6; ++j) {
              velocity(i,j) = out[j];
            }
          }
        );
        Kokkos::fence();
      }
    }

    std::cout << "Starting timed runs...\n";

    const int numTimes = 5;
    double totalTime = 0.0;
    for(int trial=0; trial<numTimes; ++trial)
    {
      // 1) Re-initialize data on device (parallel_for)
      {
        Kokkos::parallel_for("ReInitData", Kokkos::RangePolicy<>(0,N),
          KOKKOS_LAMBDA(const int i) {
            center1(i,0) = 0.0; 
            center1(i,1) = 0.0; 
            center1(i,2) = 0.0;
            center2(i,0) = 2.0; 
            center2(i,1) = 1.0; 
            center2(i,2) = 3.0;
            force2(i,0)  = 1.0; 
            force2(i,1)  = 2.0; 
            force2(i,2)  = -6.0;
            torque2(i,0) = 0.0; 
            torque2(i,1) = 0.0; 
            torque2(i,2) = 1.0;
          }
        );
        Kokkos::fence();
      }

      // 2) Measure kernel
      Kokkos::Timer timer;

      Kokkos::parallel_for("MainKernel", Kokkos::RangePolicy<>(0,N),
        KOKKOS_LAMBDA(const int i){
          double c1[3], c2[3], f2[3], t2[3], out[6];
          for(int j=0; j<3; ++j){
            c1[j] = center1(i,j);
            c2[j] = center2(i,j);
            f2[j] = force2(i,j);
            t2[j] = torque2(i,j);
          }
          computeRPYFarFieldVelocity(c1, c2, f2, t2, out);

          for(int j=0; j<6; ++j){
            velocity(i,j) = out[j];
          }
        }
      );
      Kokkos::fence();

      totalTime += timer.seconds();

        // 3) Verify results, move data to cpu and print first result
        {
          auto velocity_h = Kokkos::create_mirror_view(velocity);
          Kokkos::deep_copy(velocity_h, velocity);

          std::cout << velocity_h(0,5) << " ";
          std::cout << "\n";
        }

    }
    double callsPerSecond = (numTimes*N / totalTime)/ 1.0e6;
    std::cout << "Average time (s): " << totalTime/numTimes << "\n"
              << "Average calls/s : " << callsPerSecond << "\n";

  }
  Kokkos::finalize();
  return 0;
}