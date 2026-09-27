#pragma once

namespace ninfer::ops {

/**
 * Multiprocessor count of the active CUDA device, queried once and cached.
 *
 * Persistent-grid launchers size one resident wave from this value. Sizing from
 * a hardcoded reference-part count leaves multiprocessors idle on a device with
 * more SMs (RTX PRO 6000: 188) or oversubscribes one with fewer (GB10: 48).
 *
 * Returns the reference RTX 5090 count if the device query fails, so a launcher
 * always receives a positive, usable value.
 */
int device_sm_count();

} // namespace ninfer::ops
