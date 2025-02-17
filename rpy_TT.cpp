#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>

/**
 * Pasted from Mundyscratch. @bryce
 */

// A small tolerance used in the original code
static const double DOUBLE_ZERO = 1e-12;


inline void rpy_pairwise(
    const std::array<double, 3>& targetPos,
    const std::array<double, 3>& sourcePos,
    const std::array<double, 3>& sourceForce,
    double radius,
    double viscosity,
    double& vx, double& vy, double& vz)
{
    // Pull out coordinates and forces for convenience
    double dx = targetPos[0] - sourcePos[0];
    double dy = targetPos[1] - sourcePos[1];
    double dz = targetPos[2] - sourcePos[2];
    double fx = sourceForce[0];
    double fy = sourceForce[1];
    double fz = sourceForce[2];

    // Distance and the usual RPY factors
    double r2   = dx*dx + dy*dy + dz*dz;
    double r    = (r2 < DOUBLE_ZERO) ? 1.0 : std::sqrt(r2);
    double rinv = (r2 < DOUBLE_ZERO) ? 0.0 : (1.0 / r);

    double rinv3 = rinv * rinv * rinv;
    double rinv5 = rinv3 * rinv * rinv;

    double fdotr = fx*dx + fy*dy + fz*dz;

    // Some constants from the RPY formula
    static const double PI = 3.14159265358979323846;
    double scale_factor   = 1.0 / (8.0 * PI * viscosity);
    double a2_over_three  = (1.0/3.0) * radius * radius;

    // Terms
    double three_fdotr_rinv5 = 3.0 * fdotr * rinv5;
    double cx = fx*rinv3 - three_fdotr_rinv5 * dx;
    double cy = fy*rinv3 - three_fdotr_rinv5 * dy;
    double cz = fz*rinv3 - three_fdotr_rinv5 * dz;

    double fdotr_rinv3 = fdotr * rinv3;

    // "Leading" part of the velocity
    double v0 = scale_factor * (fx * rinv + dx * fdotr_rinv3 + a2_over_three * cx);
    double v1 = scale_factor * (fy * rinv + dy * fdotr_rinv3 + a2_over_three * cy);
    double v2 = scale_factor * (fz * rinv + dz * fdotr_rinv3 + a2_over_three * cz);

    // Laplacian correction terms
    double lap0 = 2.0 * scale_factor * cx;
    double lap1 = 2.0 * scale_factor * cy;
    double lap2 = 2.0 * scale_factor * cz;

    double lap_coeff = 0.5 * a2_over_three; 

    // Accumulate into vx, vy, vz
    vx += v0 + lap_coeff * lap0;
    vy += v1 + lap_coeff * lap1;
    vz += v2 + lap_coeff * lap2;
}

/**
 * Given 3 spheres (positions and forces), compute the resulting velocities
 * using the RPY formula. We just do a triple loop over all targets x sources.
 */
void compute_rpy_for_3_spheres(
    const std::array<std::array<double, 3>, 3>& positions,
    const std::array<std::array<double, 3>, 3>& forces,
    double radius,
    double viscosity,
    std::array<std::array<double, 3>, 3>& velocities)
{
    // Zero out all velocities first
    for (auto &vel : velocities) {
        vel = {0.0, 0.0, 0.0};
    }

    // For each target sphere, sum contributions from each source sphere
    for (int t = 0; t < 3; t++) {
        for (int s = 0; s < 3; s++) {
            if(t == s) continue;
            rpy_pairwise(positions[t], positions[s], forces[s],
                         radius, viscosity,
                         velocities[t][0], velocities[t][1], velocities[t][2]);
        }
    }
}


int main()
{
    // These are the same test data from the original code
    std::vector<double> s      = {2.10, 2.50, 3.00, 4.00, 6.00};
    std::vector<double> U2     = {0.59718, 0.49545, 0.41694, 0.31859, 0.21586};
    std::vector<double> U3     = {0.03517, 0.07393, 0.07824, 0.06925, 0.05078};

    // Viscosity and sphere radius
    double viscosity = 1.0;
    double radius    = 1.0;

    std::cout << std::fixed << std::setprecision(5);

    // We'll loop over each separation distance s[i]
    for (size_t i = 0; i < s.size(); i++)
    {
        double s_current = s[i];
        double U2_expected = -U2[i];
        double U3_expected = -U3[i];

        std::array<std::array<double, 3>, 3> positions;
        auto D = s_current * radius;
        positions[0] = {0.0,   0.0, 0.0};
        positions[1] = {D/2.0, 0.0, std::sqrt(3.0) * D/2.0};
        positions[2] = {D, 0.0, 0.0};

        // we only apply a force on sphere1 in the -z direction.
        std::array<std::array<double, 3>, 3> forces;
        forces[0] = {0.0, 0.0, 0.0}; // Sphere0 has force in -y
        forces[1] = {0.0, 0.0, -6.0 * M_PI * radius};
        forces[2] = {0.0, 0.0, 0.0};

        // We'll compute velocities
        std::array<std::array<double, 3>, 3> velocities;
        compute_rpy_for_3_spheres(positions, forces, radius, viscosity, velocities);


        double U2_computed = velocities[0][2]; // U2 is the z-component of velocity
        double U3_computed = velocities[0][0]; // U3 is the x-component of velocity

        // Print them out vs. expected
        double relErrU2 = std::fabs(U2_computed - U2_expected) / std::fabs(U2_expected) * 100.0;
        double relErrU3 = std::fabs(U3_computed - U3_expected) / std::fabs(U3_expected) * 100.0;

        std::cout << "s = " << s_current << "\n"
                  << "  U2: " << U2_computed
                  << " | expected: " << U2_expected
                  << " | % rel err: " << relErrU2 << "\n"
                  << "  U3: " << U3_computed
                  << " | expected: " << U3_expected
                  << " | % rel err: " << relErrU3 << "\n\n";
    }

    return 0;
}
