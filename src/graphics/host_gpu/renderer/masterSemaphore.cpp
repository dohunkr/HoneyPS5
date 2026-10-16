#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		EXIT("MasterSemaphore counter query failed: %s (%d), known_gpu_tick=%llu\n",
		     vk::to_string(result).c_str(), static_cast<int>(result),
		     static_cast<unsigned long long>(m_gpu_tick.load(std::memory_order_acquire)));
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	// UFC 5 (PPSA03541, Frostbite) stalls on a full-screen 4K compute dispatch during
	// the title-screen animation: tick 157401 never completes while gpu_tick stays at
	// 157400, and vkWaitSemaphores(UINT64_MAX) blocks forever before aborting with a
	// generic "Not implemented". Wait in bounded slices so a merely slow GPU (large
	// 88k-word kernels can take seconds on mid-range cards) still makes progress,
	// while a genuine hang aborts with the exact tick/counter needed for debugging.
	constexpr uint64_t kSliceNs   = 5000000000ULL;
	constexpr int      kMaxSlices = 6;
	for (int attempt = 1; attempt <= kMaxSlices; ++attempt) {
		const auto result = m_graphics.device.waitSemaphores(&wait_info, kSliceNs);
		if (result == vk::Result::eSuccess) {
			Refresh();
			return;
		}
		Refresh();
		if (IsFree(tick)) {
			return;
		}
		uint64_t   counter = 0;
		const auto counter_result =
		    m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
		const auto         known      = m_gpu_tick.load(std::memory_order_acquire);
		constexpr uint64_t kTimeoutNs = kSliceNs;
		LOGF("MasterSemaphore::Wait: %s (%d) after %lluns, tick=%llu gpu_tick=%llu "
		     "counter=%llu attempt=%d/%d (UFC 5 title-screen hang: tick 157401/gpu 157400)\n",
		     vk::to_string(result).c_str(), static_cast<int>(result),
		     static_cast<unsigned long long>(kTimeoutNs), static_cast<unsigned long long>(tick),
		     static_cast<unsigned long long>(known), static_cast<unsigned long long>(counter),
		     attempt, kMaxSlices);
		if (result == vk::Result::eErrorDeviceLost ||
		    counter_result == vk::Result::eErrorDeviceLost) {
			EXIT("MasterSemaphore GPU hang (device lost): tick=%llu gpu_tick=%llu counter=%llu\n",
			     static_cast<unsigned long long>(tick), static_cast<unsigned long long>(known),
			     static_cast<unsigned long long>(counter));
		}
		if (result != vk::Result::eTimeout) {
			EXIT("MasterSemaphore wait failed: %s (%d), tick=%llu gpu_tick=%llu counter=%llu\n",
			     vk::to_string(result).c_str(), static_cast<int>(result),
			     static_cast<unsigned long long>(tick), static_cast<unsigned long long>(known),
			     static_cast<unsigned long long>(counter));
		}
	}
	uint64_t counter = 0;
	(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT("MasterSemaphore GPU stall: tick=%llu gpu_tick=%llu counter=%llu "
	     "(Frostbite full-screen compute did not complete in %ds; see issue #905)\n",
	     static_cast<unsigned long long>(tick),
	     static_cast<unsigned long long>(m_gpu_tick.load(std::memory_order_acquire)),
	     static_cast<unsigned long long>(counter), kMaxSlices * 5);
}

} // namespace Libs::Graphics
