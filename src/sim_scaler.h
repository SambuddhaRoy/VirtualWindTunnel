#pragma once
#include "vk_types.h"
#include "environment.h"

namespace vwt {

class SimulationScaler {
public:
    /// Map physical velocity to lattice velocity based on Mach number constraints.
    static float toLatticeVelocity(float physicalVel, float speedOfSound) {
        // LBM speed of sound c_s = 1/sqrt(3) ~ 0.577.
        // We want to keep Ma < 0.3 for stability.
        // latticeVel / c_s = physicalVel / physicalSoundSpeed
        // latticeVel = (physicalVel / physicalSoundSpeed) * (1.0 / sqrt(3))
        return (physicalVel / speedOfSound) * 0.57735f;
    }

    /// Calculate LBM relaxation time (tau) from physical kinematic viscosity.
    static float calculateTau(float kinematicViscosity, float latticeDx, float latticeDt) {
        // nu_lb = nu_phys * dt / dx^2
        float nuLB = kinematicViscosity * (latticeDt / (latticeDx * latticeDx));
        return 3.0f * nuLB + 0.5f;
    }

    /// Helper to find a stable latticeDt given a desired tau range.
    static float suggestLatticeDt(float kinematicViscosity, float latticeDx, float targetTau = 0.6f) {
        float targetNuLB = (targetTau - 0.5f) / 3.0f;
        return targetNuLB * (latticeDx * latticeDx) / kinematicViscosity;
    }
};

} // namespace vwt
