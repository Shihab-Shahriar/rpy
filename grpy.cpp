#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>

using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using Matrix6x6 = std::array<std::array<double, 6>, 6>;

namespace {

Matrix3x3 identityMatrix() {
    Matrix3x3 I{};
    for (int i = 0; i < 3; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

Matrix3x3 outerProduct(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    Matrix3x3 result{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

Matrix3x3 epsilonMatrix(const std::array<double, 3>& r) {
    Matrix3x3 e{};
    e[0][0] = 0.0;   e[0][1] = -r[2]; e[0][2] =  r[1];
    e[1][0] =  r[2]; e[1][1] = 0.0;   e[1][2] = -r[0];
    e[2][0] = -r[1]; e[2][1] =  r[0]; e[2][2] = 0.0;
    return e;
}

Matrix3x3 addMatrices(const Matrix3x3& a, const Matrix3x3& b) {
    Matrix3x3 result{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

Matrix3x3 scaleMatrix(const Matrix3x3& m, double scalar) {
    Matrix3x3 result{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = m[i][j] * scalar;
        }
    }
    return result;
}

} // namespace

Matrix6x6 computeRPYFarFieldInteraction(const std::array<double, 3>& center1,
                                        const std::array<double, 3>& center2) {
    const double a = 1.0; // Constant radius
    const double eta = 1.0; // Viscosity (scales with coefficients)

    std::array<double, 3> dr = {
        center1[0] - center2[0],
        center1[1] - center2[1],
        center1[2] - center2[2]
    };

    double r_sq = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
    double r = std::sqrt(r_sq);
    
    if (r <= 2.0 * a) {
        return {}; // Return zero matrix if not in far field
    }

    std::array<double, 3> r_hat = {dr[0]/r, dr[1]/r, dr[2]/r};

    double pi = M_PI;
    double r3 = r * r * r;
    double r5 = r3 * r * r;

    // Coefficients for muTT
    double TT_identity = (1.0 / (8.0 * pi * r)) * (1.0 + (2.0 * a*a) / (3.0 * r_sq));
    double TT_rhat = (1.0 / (8.0 * pi * r)) * (1.0 - (2.0 * a*a) / r_sq);

    // Coefficients for muRR
    double RR_identity = -1.0 / (16.0 * pi * r3);
    double RR_rhat = 3.0 / (16.0 * pi * r3);

    // Coefficient for muRT
    double RT_scale = 1.0 / (8.0 * pi * r_sq);

    // Construct muTT: TT_identity * I + TT_rhat * r_hat ⊗ r_hat
    Matrix3x3 I = identityMatrix();
    Matrix3x3 muTT = addMatrices(scaleMatrix(I, TT_identity),
                                 scaleMatrix(outerProduct(r_hat, r_hat), TT_rhat));

    // Construct muRR: RR_identity * I + RR_rhat * r_hat ⊗ r_hat
    Matrix3x3 muRR = addMatrices(scaleMatrix(I, RR_identity),
                                 scaleMatrix(outerProduct(r_hat, r_hat), RR_rhat));

    // Construct muRT: RT_scale * epsilon(r_hat)
    Matrix3x3 muRT = scaleMatrix(epsilonMatrix(r_hat), RT_scale);

    // Transpose of muRT for upper-right block
    Matrix3x3 muRT_transposed{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            muRT_transposed[i][j] = muRT[j][i];
        }
    }

    // Assemble 6x6 mobility matrix
    Matrix6x6 M{};
    // muTT (top-left 3x3)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            M[i][j] = muTT[i][j];
        }
    }
    // muRT^T (top-right 3x3)
    for (int i = 0; i < 3; ++i) {
        for (int j = 3; j < 6; ++j) {
            M[i][j] = muRT_transposed[i][j-3];
        }
    }
    // muRT (bottom-left 3x3)
    for (int i = 3; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            M[i][j] = muRT[i-3][j];
        }
    }
    // muRR (bottom-right 3x3)
    for (int i = 3; i < 6; ++i) {
        for (int j = 3; j < 6; ++j) {
            M[i][j] = muRR[i-3][j-3];
        }
    }

    return M;
}

void compute_rpy_for_3_spheres(
    const std::array<std::array<double, 3>, 3>& positions,
    const std::array<std::array<double, 3>, 3>& forces,
    const std::array<std::array<double, 3>, 3>& torques,
    double radius,
    double viscosity,
    std::array<std::array<double, 3>, 3>& velocities,
    std::array<std::array<double, 3>, 3>& angular_velocities)
{
    for (auto &vel : velocities) vel = {0.0, 0.0, 0.0};
    for (auto &ang : angular_velocities) ang = {0.0, 0.0, 0.0};

    const double mu_tt = 1.0 / (6.0 * M_PI * viscosity * radius);
    const double mu_rr = 1.0 / (8.0 * M_PI * viscosity * radius * radius * radius);

    for (int t = 0; t < 3; t++) {
        for (int s = 0; s < 3; s++) {
            Matrix6x6 M = computeRPYFarFieldInteraction(positions[t], positions[s]);

            // Accumulate velocity from force and torque of source s
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    velocities[t][i] += M[i][j] * forces[s][j];
                    velocities[t][i] += M[i][j + 3] * torques[s][j];
                }
            }

            // Accumulate angular velocity from force and torque of source s
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    angular_velocities[t][i] += M[i + 3][j] * forces[s][j];
                    angular_velocities[t][i] += M[i + 3][j + 3] * torques[s][j];
                }
            }
        }
    }
}

int main() {
    std::vector<double> s = {2.10, 2.50, 3.00, 4.00, 6.00};
    std::vector<double> U2 = {0.59718, 0.49545, 0.41694, 0.31859, 0.21586};
    std::vector<double> U3 = {0.03517, 0.07393, 0.07824, 0.06925, 0.05078};
    std::vector<double> Omega3 = {0.052035, 0.045466, 0.035022, 0.021634, 0.010159};
    
    const double viscosity = 1.0;
    const double radius = 1.0;

    std::cout << std::fixed << std::setprecision(5);

    for (size_t i = 0; i < s.size(); i++) {
        double s_current = s[i];
        double U2_expected = -U2[i];
        double U3_expected = -U3[i];
        double Omega3_expected = Omega3[i];

        std::array<std::array<double, 3>, 3> positions;
        double D = s_current * radius;
        positions[0] = {0.0, 0.0, 0.0};
        positions[1] = {D/2.0, 0.0, std::sqrt(3.0) * D/2.0};
        positions[2] = {D, 0.0, 0.0};

        std::array<std::array<double, 3>, 3> forces;
        forces[0] = {0.0, 0.0, 0.0};
        forces[1] = {0.0, 0.0, -6.0 * M_PI * radius}; // Force in -z direction
        forces[2] = {0.0, 0.0, 0.0};

        std::array<std::array<double, 3>, 3> torques = {};
        
        std::array<std::array<double, 3>, 3> velocities;
        std::array<std::array<double, 3>, 3> angular_velocities;
        compute_rpy_for_3_spheres(positions, forces, torques, radius, viscosity, velocities, angular_velocities);

        double U2_computed = velocities[0][2]; // z-component (U2)
        double U3_computed = velocities[0][0]; // x-component (U3)
        double Omega3_computed = angular_velocities[0][1]; // y-component (Omega3)

        double relErrU2 = std::fabs(U2_computed - U2_expected) / std::fabs(U2_expected) * 100.0;
        double relErrU3 = std::fabs(U3_computed - U3_expected) / std::fabs(U3_expected) * 100.0;
        double relErrOmega3 = std::fabs(Omega3_computed - Omega3_expected) / std::fabs(Omega3_expected) * 100.0;

        std::cout << "s = " << s_current << "\n"
                  << "  U2: " << U2_computed << " | expected: " << U2_expected
                  << " | % rel err: " << relErrU2 << "\n"
                  << "  U3: " << U3_computed << " | expected: " << U3_expected
                  << " | % rel err: " << relErrU3 << "\n"
                  << "  Omega3: " << Omega3_computed << " | expected: " << Omega3_expected
                  << " | % rel err: " << relErrOmega3 << "\n\n";
    }

    return 0;
}