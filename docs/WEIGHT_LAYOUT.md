# Weight layout

The default policy is direct consumption of the packed source representation. The inspector currently records
source shape, logical shape, byte offset, alignment, storage dtype, and scale relationships.

No device-side scale swizzle or weight transformation has been selected. Any future in-memory transformation must
preserve quantized values and scales exactly, stream into the final allocation, avoid a second persistent copy, and
include load-time and peak-memory measurements.

