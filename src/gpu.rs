//! Experimental GPU backend for the bunching-effect terminal evaluation.
//!
//! **This module is a spike.** It is not yet wired into the solver; it exists to measure
//! whether offloading the bunching evaluation kernel to a GPU is worthwhile, and to establish
//! the runtime-detection and fallback pattern such an integration would use.
//!
//! # Availability
//!
//! Compiled only with the `gpu` feature, which is off by default. Even when compiled in, the
//! GPU is never mandatory: [`GpuEvaluator::try_new`] probes for a compatible device at runtime
//! and returns `None` if there is none, if the backend fails to initialise, or if the device
//! cannot satisfy the buffer sizes required. Callers fall back to the CPU path, which remains
//! the reference implementation.
//!
//! # Kernel
//!
//! The offloaded operation is the batched conditional inner product performed at bunching
//! terminal nodes (see `PostFlopGame::evaluate_internal_bunching`). For each (node, player
//! hand) pair it computes a dot product of the opponent reach probabilities against a row of
//! the precomputed combination table, with each term scaled by a win/lose/tie amount selected
//! from a hand-strength comparison.

use crate::sliceop::inner_product_cond;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Sentinel matching `index == 0` ("no valid combination") in the CPU implementation.
const SKIP: u32 = u32::MAX;

/// Workgroup size; must match `@workgroup_size` in `gpu.wgsl`.
const WORKGROUP_SIZE: u32 = 256;

/// One batch of bunching terminal evaluations.
///
/// A batch groups several terminal nodes so that a single GPU dispatch carries enough work to
/// amortise its launch overhead — evaluating one node at a time would be dominated by it.
pub struct EvalBatch<'a> {
    /// The precomputed combination table. Constant for the lifetime of a solve, so it is
    /// uploaded to the GPU once and reused across dispatches.
    pub arena: &'a [f32],
    /// Opponent reach probabilities, `len` values per node.
    pub cfreach: &'a [f32],
    /// Opponent hand strengths, `len` values per node.
    pub cond: &'a [u16],
    /// Start offset into `arena` for each output, or [`SKIP`] where the CPU code would see
    /// `index == 0` and write `0.0`.
    pub offsets: &'a [u32],
    /// Dot product length (the opponent's hand count).
    pub len: usize,
    /// Outputs per node (the player's hand count).
    pub num_rows: usize,
    pub threshold: u16,
    pub less: f32,
    pub greater: f32,
    pub equal: f32,
}

impl EvalBatch<'_> {
    fn num_nodes(&self) -> usize {
        self.offsets.len().checked_div(self.num_rows).unwrap_or(0)
    }
}

/// Reference CPU implementation, using the same `inner_product_cond` the solver uses.
///
/// Parallelised across outputs so the comparison against the GPU reflects how the solver
/// actually runs (all cores busy), rather than a single-threaded straw man.
pub fn cpu_batched(batch: &EvalBatch) -> Vec<f32> {
    let eval = |(o, &base): (usize, &u32)| -> f32 {
        if base == SKIP {
            return 0.0;
        }
        let node = o / batch.num_rows;
        let cf = &batch.cfreach[node * batch.len..(node + 1) * batch.len];
        let cond = &batch.cond[node * batch.len..(node + 1) * batch.len];
        let row = &batch.arena[base as usize..base as usize + batch.len];
        inner_product_cond(
            cf,
            row,
            cond,
            batch.threshold,
            batch.less,
            batch.greater,
            batch.equal,
        )
    };

    #[cfg(feature = "rayon")]
    {
        batch.offsets.par_iter().enumerate().map(eval).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        batch.offsets.iter().enumerate().map(eval).collect()
    }
}

/// Branchless CPU variant, accumulating in `f32`.
///
/// The reference kernel carries a comment noting that its `match` prevents vectorisation, and
/// it widens every product to `f64`. Both suppress SIMD. This variant replaces the comparison
/// chain with arithmetic selection and keeps the accumulator in `f32`, so it is included in
/// the benchmark to establish how much of any GPU win is really just untapped CPU headroom.
pub fn cpu_batched_branchless(batch: &EvalBatch) -> Vec<f32> {
    let eval = |(o, &base): (usize, &u32)| -> f32 {
        if base == SKIP {
            return 0.0;
        }
        let node = o / batch.num_rows;
        let cf = &batch.cfreach[node * batch.len..(node + 1) * batch.len];
        let cond = &batch.cond[node * batch.len..(node + 1) * batch.len];
        let row = &batch.arena[base as usize..base as usize + batch.len];

        const CHUNK: usize = 8;
        let mut acc = [0.0f32; CHUNK];
        let chunks = batch.len / CHUNK * CHUNK;

        for i in (0..chunks).step_by(CHUNK) {
            for j in 0..CHUNK {
                unsafe {
                    let c = *cond.get_unchecked(i + j);
                    let lt = (c < batch.threshold) as u32 as f32;
                    let gt = (c > batch.threshold) as u32 as f32;
                    let z = lt * batch.less + gt * batch.greater + (1.0 - lt - gt) * batch.equal;
                    *acc.get_unchecked_mut(j) +=
                        *cf.get_unchecked(i + j) * *row.get_unchecked(i + j) * z;
                }
            }
        }
        let mut tail = 0.0f32;
        for i in chunks..batch.len {
            let c = cond[i];
            let lt = (c < batch.threshold) as u32 as f32;
            let gt = (c > batch.threshold) as u32 as f32;
            let z = lt * batch.less + gt * batch.greater + (1.0 - lt - gt) * batch.equal;
            tail += cf[i] * row[i] * z;
        }
        acc.iter().sum::<f32>() + tail
    };

    #[cfg(feature = "rayon")]
    {
        batch.offsets.par_iter().enumerate().map(eval).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        batch.offsets.iter().enumerate().map(eval).collect()
    }
}

// ---------------------------------------------------------------------------------------
// GPU path
// ---------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    len: u32,
    num_rows: u32,
    threshold: u32,
    _pad0: u32,
    less: f32,
    greater: f32,
    equal: f32,
    _pad1: f32,
}

/// A GPU device plus the compiled bunching-evaluation pipeline.
///
/// Obtained through [`GpuEvaluator::try_new`], which never panics and never requires a GPU to
/// be present.
pub struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    info: String,
}

impl GpuEvaluator {
    /// Probes for a usable GPU and compiles the kernel.
    ///
    /// Returns `None` when no compatible adapter exists, when device creation fails, or when
    /// the backend cannot be initialised at all. Callers must treat `None` as "use the CPU
    /// path" rather than as an error.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .ok()?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("postflop-solver bunching evaluator"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            ..Default::default()
        }))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bunching_eval"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bunching_eval_layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bunching_eval_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            layout,
            info: format!("{:?} / {:?}", adapter_info.name, adapter_info.backend),
        })
    }

    /// Human-readable description of the selected adapter.
    pub fn adapter_info(&self) -> &str {
        &self.info
    }

    /// Uploads the (solve-constant) combination table once, returning a handle to reuse.
    pub fn upload_arena(&self, arena: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bunching_arena"),
                contents: bytemuck::cast_slice(arena),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Builds all buffers and the bind group for a batch, without running or reading back.
    ///
    /// wgpu resources are reference counted, so the returned bind group keeps its buffers
    /// alive on its own.
    fn prepare(&self, batch: &EvalBatch, arena_buf: &wgpu::Buffer) -> wgpu::BindGroup {
        use wgpu::util::DeviceExt;

        let num_nodes = batch.num_nodes();
        let packed_len = batch.len.div_ceil(2);
        let mut packed = vec![0u32; num_nodes * packed_len];
        for n in 0..num_nodes {
            let src = &batch.cond[n * batch.len..(n + 1) * batch.len];
            let dst = &mut packed[n * packed_len..(n + 1) * packed_len];
            for (i, &c) in src.iter().enumerate() {
                if i % 2 == 0 {
                    dst[i / 2] |= c as u32;
                } else {
                    dst[i / 2] |= (c as u32) << 16;
                }
            }
        }

        let mk = |label: &str, data: &[u8], usage: wgpu::BufferUsages| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage,
                })
        };

        let st = wgpu::BufferUsages::STORAGE;
        let cfreach_buf = mk("cfreach", bytemuck::cast_slice(batch.cfreach), st);
        let cond_buf = mk("cond", bytemuck::cast_slice(&packed), st);
        let offsets_buf = mk("offsets", bytemuck::cast_slice(batch.offsets), st);

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (batch.offsets.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params {
            len: batch.len as u32,
            num_rows: batch.num_rows as u32,
            threshold: batch.threshold as u32,
            _pad0: 0,
            less: batch.less,
            greater: batch.greater,
            equal: batch.equal,
            _pad1: 0.0,
        };
        let params_buf = mk(
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cfreach_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cond_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatches the kernel `repeats` times against pre-built buffers and returns the mean
    /// wall-clock time per dispatch.
    ///
    /// This isolates raw kernel throughput from the per-call buffer construction and readback
    /// that [`Self::eval_batch`] performs, i.e. it measures the ceiling a fully optimised
    /// integration could approach.
    pub fn time_compute_only(
        &self,
        batch: &EvalBatch,
        arena_buf: &wgpu::Buffer,
        repeats: u32,
    ) -> f64 {
        let prepared = self.prepare(batch, arena_buf);
        // Warm up.
        self.dispatch(&prepared, batch.offsets.len() as u32, 2);
        let t = std::time::Instant::now();
        self.dispatch(&prepared, batch.offsets.len() as u32, repeats);
        t.elapsed().as_secs_f64() / repeats as f64
    }

    fn dispatch(&self, bind_group: &wgpu::BindGroup, num_out: u32, repeats: u32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..repeats {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_out, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    /// Evaluates one batch on the GPU.
    pub fn eval_batch(&self, batch: &EvalBatch, arena_buf: &wgpu::Buffer) -> Vec<f32> {
        use wgpu::util::DeviceExt;

        let num_out = batch.offsets.len();
        let num_nodes = batch.num_nodes();

        // Pack u16 conditions two per u32.
        let packed_len = batch.len.div_ceil(2);
        let mut packed = vec![0u32; num_nodes * packed_len];
        for n in 0..num_nodes {
            let src = &batch.cond[n * batch.len..(n + 1) * batch.len];
            let dst = &mut packed[n * packed_len..(n + 1) * packed_len];
            for (i, &c) in src.iter().enumerate() {
                if i % 2 == 0 {
                    dst[i / 2] |= c as u32;
                } else {
                    dst[i / 2] |= (c as u32) << 16;
                }
            }
        }

        let mk = |label: &str, data: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let cfreach_buf = mk("cfreach", bytemuck::cast_slice(batch.cfreach));
        let cond_buf = mk("cond", bytemuck::cast_slice(&packed));
        let offsets_buf = mk("offsets", bytemuck::cast_slice(batch.offsets));

        let out_size = (num_out * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params {
            len: batch.len as u32,
            num_rows: batch.num_rows as u32,
            threshold: batch.threshold as u32,
            _pad0: 0,
            less: batch.less,
            greater: batch.greater,
            equal: batch.equal,
            _pad1: 0.0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cfreach_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cond_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_out as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice.get_mapped_range().expect("map failed");
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

/// Silences an unused-constant warning when the GPU path is compiled but unused.
const _: u32 = WORKGROUP_SIZE;
