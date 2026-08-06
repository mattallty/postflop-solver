use super::*;
use crate::interface::*;
use std::ptr;
use std::slice;

/// Builds a storage slice, tolerating an unassigned pointer.
///
/// `len` is fixed at tree-build time, but the storage pointers are only assigned while memory
/// is allocated: they are null before [`PostFlopGame::allocate_memory`] and after
/// [`PostFlopGame::free_memory`]. The accessors below are safe and publicly reachable through
/// the [`GameNode`] trait, so in that state they must return an empty slice rather than
/// fabricate one that no allocation backs.
#[inline]
fn storage_slice<'a, T>(ptr: *const u8, len: usize) -> &'a [T] {
    if ptr.is_null() {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ptr as *const T, len) }
    }
}

/// Mutable counterpart of [`storage_slice`].
#[inline]
fn storage_slice_mut<'a, T>(ptr: *mut u8, len: usize) -> &'a mut [T] {
    if ptr.is_null() {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(ptr as *mut T, len) }
    }
}

impl GameNode for PostFlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player & PLAYER_CHANCE_FLAG != 0
    }

    #[inline]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        let prev_player = self.player & PLAYER_MASK;
        match prev_player {
            0 => Some(1),
            1 => Some(0),
            _ => None,
        }
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<'_, Self> {
        self.children()[action].lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        storage_slice::<f32>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        storage_slice_mut::<f32>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        storage_slice::<f32>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        storage_slice_mut::<f32>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        storage_slice::<f32>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        storage_slice_mut::<f32>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn has_cfvalues_ip(&self) -> bool {
        self.num_elements_ip != 0
    }

    #[inline]
    fn cfvalues_ip(&self) -> &[f32] {
        storage_slice::<f32>(self.storage3, self.num_elements_ip as usize)
    }

    #[inline]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        storage_slice_mut::<f32>(self.storage3, self.num_elements_ip as usize)
    }

    #[inline]
    fn cfvalues_chance(&self) -> &[f32] {
        storage_slice::<f32>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        storage_slice_mut::<f32>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        storage_slice::<u16>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        storage_slice_mut::<u16>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn regrets_compressed(&self) -> &[i16] {
        storage_slice::<i16>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        storage_slice_mut::<i16>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues_compressed(&self) -> &[i16] {
        storage_slice::<i16>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        storage_slice_mut::<i16>(self.storage2, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        storage_slice::<i16>(self.storage3, self.num_elements_ip as usize)
    }

    #[inline]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        storage_slice_mut::<i16>(self.storage3, self.num_elements_ip as usize)
    }

    #[inline]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        storage_slice::<i16>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        storage_slice_mut::<i16>(self.storage1, self.num_elements as usize)
    }

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn regret_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_regret_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_cfvalue_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_ip_scale(&self) -> f32 {
        self.scale3
    }

    #[inline]
    fn set_cfvalue_ip_scale(&mut self, scale: f32) {
        self.scale3 = scale;
    }

    #[inline]
    fn cfvalue_chance_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_cfvalue_chance_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        self.river == NOT_DEALT
    }
}

impl Default for PostFlopNode {
    #[inline]
    fn default() -> Self {
        Self {
            prev_action: Action::None,
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            is_locked: false,
            amount: 0,
            children_offset: 0,
            num_children: 0,
            num_elements_ip: 0,
            storage1: ptr::null_mut(),
            storage2: ptr::null_mut(),
            storage3: ptr::null_mut(),
            num_elements: 0,
            scale1: 0.0,
            scale2: 0.0,
            scale3: 0.0,
        }
    }
}

impl PostFlopNode {
    #[inline]
    pub(super) fn children(&self) -> &[MutexLike<Self>] {
        // This is safe because `MutexLike<T>` is a `repr(transparent)` wrapper around `T`.
        let self_ptr = self as *const _ as *const MutexLike<PostFlopNode>;
        unsafe {
            slice::from_raw_parts(
                self_ptr.add(self.children_offset as usize),
                self.num_children as usize,
            )
        }
    }
}
