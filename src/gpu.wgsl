// Batched conditional inner product: the bunching-effect terminal evaluation kernel.
//
// For every output element `o` (one per (node, player hand) pair) this computes
//     out[o] = sum_i cfreach[node][i] * arena[offsets[o] + i] * z(cond[node][i])
// where z selects `less` / `greater` / `equal` by comparing cond[i] against `threshold`.
// This mirrors `inner_product_cond` in src/sliceop.rs.
//
// One workgroup per output element; threads stride over the dot product and then perform a
// tree reduction in workgroup memory. The tree reduction also gives better error growth
// (O(log n)) than the CPU path's sequential accumulation, which is why f32 suffices here
// even though the CPU version accumulates in f64.

struct Params {
    len: u32,
    num_rows: u32,
    threshold: u32,
    _pad0: u32,
    less: f32,
    greater: f32,
    equal: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read> arena: array<f32>;
@group(0) @binding(1) var<storage, read> cfreach: array<f32>;
@group(0) @binding(2) var<storage, read> cond: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 256u;
const SKIP: u32 = 4294967295u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_index = wid.x;
    let tid = lid.x;
    let len = params.len;
    let node = out_index / params.num_rows;
    let base = offsets[out_index];

    var acc = 0.0;

    // `index == 0` in the Rust code means "no valid combination"; encoded as SKIP here.
    if (base != SKIP) {
        let cf_base = node * len;
        let cond_base = node * ((len + 1u) / 2u);
        var i = tid;
        loop {
            if (i >= len) { break; }

            let x = cfreach[cf_base + i];
            let y = arena[base + i];

            // u16 conditions are packed two per u32.
            let packed = cond[cond_base + (i >> 1u)];
            var c: u32;
            if ((i & 1u) == 0u) {
                c = packed & 65535u;
            } else {
                c = packed >> 16u;
            }

            var z: f32;
            if (c < params.threshold) {
                z = params.less;
            } else if (c > params.threshold) {
                z = params.greater;
            } else {
                z = params.equal;
            }

            acc = acc + x * y * z;
            i = i + WG;
        }
    }

    partial[tid] = acc;
    workgroupBarrier();

    // Tree reduction. The barrier stays in uniform control flow.
    var s = WG >> 1u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            partial[tid] = partial[tid] + partial[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out[out_index] = partial[0];
    }
}
